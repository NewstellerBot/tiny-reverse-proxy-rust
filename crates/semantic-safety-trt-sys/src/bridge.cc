#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

extern "C" {

struct ss_trt_status {
  int32_t code;
  char* message;
};

struct ss_trt_embedding_engine;
struct ss_trt_reranker_engine;

void ss_trt_status_free(ss_trt_status* status);
ss_trt_embedding_engine* ss_trt_embedding_engine_load(const char* engine_path, int32_t device_id,
                                                      size_t max_batch_size,
                                                      size_t max_sequence_length,
                                                      ss_trt_status* status);
void ss_trt_embedding_engine_destroy(ss_trt_embedding_engine* engine);
size_t ss_trt_embedding_engine_output_dim(const ss_trt_embedding_engine* engine);
bool ss_trt_embedding_engine_warmup(ss_trt_embedding_engine* engine, ss_trt_status* status);
bool ss_trt_embedding_engine_infer(ss_trt_embedding_engine* engine, const int32_t* input_ids,
                                   const int32_t* attention_mask, size_t batch_size,
                                   size_t sequence_length, float* output_embeddings,
                                   size_t output_len, ss_trt_status* status);

ss_trt_reranker_engine* ss_trt_reranker_engine_load(const char* engine_path, int32_t device_id,
                                                    size_t max_batch_size,
                                                    size_t max_sequence_length,
                                                    ss_trt_status* status);
void ss_trt_reranker_engine_destroy(ss_trt_reranker_engine* engine);
bool ss_trt_reranker_engine_warmup(ss_trt_reranker_engine* engine, ss_trt_status* status);
bool ss_trt_reranker_engine_infer(ss_trt_reranker_engine* engine, const int32_t* input_ids,
                                  const int32_t* attention_mask, size_t batch_size,
                                  size_t sequence_length, float* output_scores,
                                  size_t output_len, ss_trt_status* status);
}

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity > Severity::kWARNING) {
      return;
    }
    std::fprintf(stderr, "[semantic-safety-trt] %s\n", msg);
  }
};

Logger kLogger;

template <typename T>
struct Destroyer {
  void operator()(T* ptr) const {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, Destroyer<T>>;

void clear_status(ss_trt_status* status) {
  if (status == nullptr) {
    return;
  }
  if (status->message != nullptr) {
    std::free(status->message);
    status->message = nullptr;
  }
  status->code = 0;
}

void set_status(ss_trt_status* status, int32_t code, const std::string& message) {
  if (status == nullptr) {
    return;
  }
  clear_status(status);
  status->code = code;
  status->message = static_cast<char*>(std::malloc(message.size() + 1));
  if (status->message != nullptr) {
    std::memcpy(status->message, message.c_str(), message.size() + 1);
  }
}

bool read_engine_bytes(const std::string& path, std::vector<char>* out, ss_trt_status* status) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    set_status(status, 1, "failed to open TensorRT engine: " + path);
    return false;
  }
  input.seekg(0, std::ios::end);
  std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size <= 0) {
    set_status(status, 2, "TensorRT engine file is empty: " + path);
    return false;
  }
  out->resize(static_cast<size_t>(size));
  if (!input.read(out->data(), size)) {
    set_status(status, 3, "failed to read TensorRT engine: " + path);
    return false;
  }
  return true;
}

bool cuda_ok(cudaError_t result, const std::string& action, ss_trt_status* status) {
  if (result == cudaSuccess) {
    return true;
  }
  set_status(status, 4, action + ": " + cudaGetErrorString(result));
  return false;
}

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  bool allocate(size_t next_bytes, ss_trt_status* status) {
    if (ptr != nullptr && bytes >= next_bytes) {
      return true;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
    if (!cuda_ok(cudaMalloc(&ptr, next_bytes), "cudaMalloc failed", status)) {
      return false;
    }
    bytes = next_bytes;
    return true;
  }
};

struct TensorRtEngineBase {
  std::string input_ids_name = "input_ids";
  std::string attention_mask_name = "attention_mask";
  std::string output_name;
  size_t max_batch_size = 0;
  size_t max_sequence_length = 0;
  int32_t device_id = 0;
  nvinfer1::DataType input_ids_dtype = nvinfer1::DataType::kINT32;
  nvinfer1::DataType attention_mask_dtype = nvinfer1::DataType::kINT32;
  cudaStream_t stream = nullptr;
  TrtPtr<nvinfer1::IRuntime> runtime{nullptr};
  TrtPtr<nvinfer1::ICudaEngine> engine{nullptr};
  TrtPtr<nvinfer1::IExecutionContext> context{nullptr};
  DeviceBuffer input_ids_buffer;
  DeviceBuffer attention_mask_buffer;
  DeviceBuffer output_buffer;
  std::vector<int64_t> input_ids_i64;
  std::vector<int64_t> attention_mask_i64;

  virtual ~TensorRtEngineBase() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  bool init(const char* engine_path, int32_t next_device_id, size_t next_max_batch_size,
            size_t next_max_sequence_length, const char* next_output_name, ss_trt_status* status) {
    device_id = next_device_id;
    max_batch_size = next_max_batch_size;
    max_sequence_length = next_max_sequence_length;
    output_name = next_output_name;

    if (!cuda_ok(cudaSetDevice(device_id), "cudaSetDevice failed", status)) {
      return false;
    }
    if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate failed", status)) {
      return false;
    }

    std::vector<char> engine_bytes;
    if (!read_engine_bytes(engine_path, &engine_bytes, status)) {
      return false;
    }

    runtime.reset(nvinfer1::createInferRuntime(kLogger));
    if (!runtime) {
      set_status(status, 5, "failed to create TensorRT runtime");
      return false;
    }
    engine.reset(runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!engine) {
      set_status(status, 6, "failed to deserialize TensorRT engine");
      return false;
    }
    context.reset(engine->createExecutionContext());
    if (!context) {
      set_status(status, 7, "failed to create TensorRT execution context");
      return false;
    }
    input_ids_dtype = engine->getTensorDataType(input_ids_name.c_str());
    attention_mask_dtype = engine->getTensorDataType(attention_mask_name.c_str());
    if ((input_ids_dtype != nvinfer1::DataType::kINT32 &&
         input_ids_dtype != nvinfer1::DataType::kINT64) ||
        (attention_mask_dtype != nvinfer1::DataType::kINT32 &&
         attention_mask_dtype != nvinfer1::DataType::kINT64)) {
      set_status(status, 26, "semantic safety engines must expose INT32 or INT64 token inputs");
      return false;
    }
    return true;
  }

  bool set_shapes(size_t batch_size, size_t sequence_length, ss_trt_status* status) {
    if (batch_size == 0 || batch_size > max_batch_size) {
      set_status(status, 8, "batch size exceeds configured maximum");
      return false;
    }
    if (sequence_length == 0 || sequence_length > max_sequence_length) {
      set_status(status, 9, "sequence length exceeds configured maximum");
      return false;
    }
    nvinfer1::Dims dims;
    dims.nbDims = 2;
    dims.d[0] = static_cast<int32_t>(batch_size);
    dims.d[1] = static_cast<int32_t>(sequence_length);
    if (!context->setInputShape(input_ids_name.c_str(), dims)) {
      set_status(status, 10, "failed to set input_ids shape");
      return false;
    }
    if (!context->setInputShape(attention_mask_name.c_str(), dims)) {
      set_status(status, 11, "failed to set attention_mask shape");
      return false;
    }
    return true;
  }
};

bool copy_single_input(TensorRtEngineBase* engine, const char* tensor_name, const int32_t* values,
                       size_t element_count, nvinfer1::DataType dtype, DeviceBuffer* buffer,
                       std::vector<int64_t>* widened_values, ss_trt_status* status) {
  size_t tensor_bytes = 0;
  const void* host_values = values;
  if (dtype == nvinfer1::DataType::kINT64) {
    widened_values->resize(element_count);
    for (size_t idx = 0; idx < element_count; ++idx) {
      (*widened_values)[idx] = static_cast<int64_t>(values[idx]);
    }
    host_values = widened_values->data();
    tensor_bytes = element_count * sizeof(int64_t);
  } else {
    tensor_bytes = element_count * sizeof(int32_t);
  }
  if (!buffer->allocate(tensor_bytes, status)) {
    return false;
  }
  if (!cuda_ok(cudaMemcpyAsync(buffer->ptr, host_values, tensor_bytes, cudaMemcpyHostToDevice,
                               engine->stream),
               std::string("failed to copy ") + tensor_name + " to GPU", status)) {
    return false;
  }
  if (!engine->context->setTensorAddress(tensor_name, buffer->ptr)) {
    set_status(status, 12, std::string("failed to bind ") + tensor_name + " tensor");
    return false;
  }
  return true;
}

bool copy_inputs(TensorRtEngineBase* engine, const int32_t* input_ids,
                 const int32_t* attention_mask, size_t batch_size, size_t sequence_length,
                 ss_trt_status* status) {
  const size_t element_count = batch_size * sequence_length;
  if (!copy_single_input(engine, engine->input_ids_name.c_str(), input_ids, element_count,
                         engine->input_ids_dtype, &engine->input_ids_buffer,
                         &engine->input_ids_i64, status) ||
      !copy_single_input(engine, engine->attention_mask_name.c_str(), attention_mask,
                         element_count, engine->attention_mask_dtype,
                         &engine->attention_mask_buffer, &engine->attention_mask_i64, status)) {
    return false;
  }
  return true;
}

bool dims_match_count(const nvinfer1::Dims& dims, size_t* element_count) {
  if (dims.nbDims <= 0) {
    return false;
  }
  size_t total = 1;
  for (int idx = 0; idx < dims.nbDims; ++idx) {
    if (dims.d[idx] <= 0) {
      return false;
    }
    total *= static_cast<size_t>(dims.d[idx]);
  }
  *element_count = total;
  return true;
}

}  // namespace

struct ss_trt_embedding_engine : TensorRtEngineBase {
  size_t output_dim = 0;
};

struct ss_trt_reranker_engine : TensorRtEngineBase {};

extern "C" {

void ss_trt_status_free(ss_trt_status* status) { clear_status(status); }

ss_trt_embedding_engine* ss_trt_embedding_engine_load(const char* engine_path, int32_t device_id,
                                                      size_t max_batch_size,
                                                      size_t max_sequence_length,
                                                      ss_trt_status* status) {
  auto* engine = new ss_trt_embedding_engine();
  if (!engine->init(engine_path, device_id, max_batch_size, max_sequence_length, "embeddings",
                    status)) {
    delete engine;
    return nullptr;
  }
  nvinfer1::Dims dims = engine->engine->getTensorShape(engine->output_name.c_str());
  if (dims.nbDims != 2 || dims.d[1] <= 0) {
    set_status(status, 14, "embedding engine must expose output tensor `embeddings` with shape [B, D]");
    delete engine;
    return nullptr;
  }
  engine->output_dim = static_cast<size_t>(dims.d[1]);
  return engine;
}

void ss_trt_embedding_engine_destroy(ss_trt_embedding_engine* engine) { delete engine; }

size_t ss_trt_embedding_engine_output_dim(const ss_trt_embedding_engine* engine) {
  return engine == nullptr ? 0 : engine->output_dim;
}

bool ss_trt_embedding_engine_warmup(ss_trt_embedding_engine* engine, ss_trt_status* status) {
  if (engine == nullptr) {
    set_status(status, 15, "embedding engine is null");
    return false;
  }
  std::vector<int32_t> input_ids(engine->max_sequence_length, 0);
  std::vector<int32_t> attention_mask(engine->max_sequence_length, 0);
  std::vector<float> output(engine->output_dim, 0.0f);
  return ss_trt_embedding_engine_infer(engine, input_ids.data(), attention_mask.data(), 1,
                                       engine->max_sequence_length, output.data(), output.size(),
                                       status);
}

bool ss_trt_embedding_engine_infer(ss_trt_embedding_engine* engine, const int32_t* input_ids,
                                   const int32_t* attention_mask, size_t batch_size,
                                   size_t sequence_length, float* output_embeddings,
                                   size_t output_len, ss_trt_status* status) {
  if (engine == nullptr) {
    set_status(status, 16, "embedding engine is null");
    return false;
  }
  if (output_len != batch_size * engine->output_dim) {
    set_status(status, 17, "embedding output buffer has unexpected length");
    return false;
  }
  if (!engine->set_shapes(batch_size, sequence_length, status) ||
      !copy_inputs(engine, input_ids, attention_mask, batch_size, sequence_length, status)) {
    return false;
  }
  nvinfer1::Dims output_dims = engine->context->getTensorShape(engine->output_name.c_str());
  size_t actual_output_len = 0;
  if (!dims_match_count(output_dims, &actual_output_len)) {
    set_status(status, 27, "embedding engine produced an invalid output shape");
    return false;
  }
  if (actual_output_len != output_len) {
    set_status(status, 28, "embedding engine output shape does not match requested batch size");
    return false;
  }
  const size_t output_bytes = output_len * sizeof(float);
  if (!engine->output_buffer.allocate(output_bytes, status)) {
    return false;
  }
  if (!engine->context->setTensorAddress(engine->output_name.c_str(), engine->output_buffer.ptr)) {
    set_status(status, 18, "failed to bind embeddings tensor");
    return false;
  }
  if (!engine->context->enqueueV3(engine->stream)) {
    set_status(status, 19, "TensorRT enqueueV3 failed for embedding engine");
    return false;
  }
  if (!cuda_ok(cudaMemcpyAsync(output_embeddings, engine->output_buffer.ptr, output_bytes,
                               cudaMemcpyDeviceToHost, engine->stream),
               "failed to copy embeddings from GPU", status) ||
      !cuda_ok(cudaStreamSynchronize(engine->stream), "cudaStreamSynchronize failed", status)) {
    return false;
  }
  return true;
}

ss_trt_reranker_engine* ss_trt_reranker_engine_load(const char* engine_path, int32_t device_id,
                                                    size_t max_batch_size,
                                                    size_t max_sequence_length,
                                                    ss_trt_status* status) {
  auto* engine = new ss_trt_reranker_engine();
  if (!engine->init(engine_path, device_id, max_batch_size, max_sequence_length, "scores",
                    status)) {
    delete engine;
    return nullptr;
  }
  nvinfer1::Dims dims = engine->engine->getTensorShape(engine->output_name.c_str());
  if (dims.nbDims != 1) {
    set_status(status, 20, "reranker engine must expose output tensor `scores` with shape [B]");
    delete engine;
    return nullptr;
  }
  return engine;
}

void ss_trt_reranker_engine_destroy(ss_trt_reranker_engine* engine) { delete engine; }

bool ss_trt_reranker_engine_warmup(ss_trt_reranker_engine* engine, ss_trt_status* status) {
  if (engine == nullptr) {
    set_status(status, 21, "reranker engine is null");
    return false;
  }
  std::vector<int32_t> input_ids(engine->max_sequence_length, 0);
  std::vector<int32_t> attention_mask(engine->max_sequence_length, 0);
  std::vector<float> output(1, 0.0f);
  return ss_trt_reranker_engine_infer(engine, input_ids.data(), attention_mask.data(), 1,
                                      engine->max_sequence_length, output.data(), output.size(),
                                      status);
}

bool ss_trt_reranker_engine_infer(ss_trt_reranker_engine* engine, const int32_t* input_ids,
                                  const int32_t* attention_mask, size_t batch_size,
                                  size_t sequence_length, float* output_scores,
                                  size_t output_len, ss_trt_status* status) {
  if (engine == nullptr) {
    set_status(status, 22, "reranker engine is null");
    return false;
  }
  if (output_len != batch_size) {
    set_status(status, 23, "reranker output buffer has unexpected length");
    return false;
  }
  if (!engine->set_shapes(batch_size, sequence_length, status) ||
      !copy_inputs(engine, input_ids, attention_mask, batch_size, sequence_length, status)) {
    return false;
  }
  nvinfer1::Dims output_dims = engine->context->getTensorShape(engine->output_name.c_str());
  size_t actual_output_len = 0;
  if (!dims_match_count(output_dims, &actual_output_len)) {
    set_status(status, 29, "reranker engine produced an invalid output shape");
    return false;
  }
  if (actual_output_len != output_len) {
    set_status(status, 30, "reranker engine output shape does not match requested batch size");
    return false;
  }
  const size_t output_bytes = output_len * sizeof(float);
  if (!engine->output_buffer.allocate(output_bytes, status)) {
    return false;
  }
  if (!engine->context->setTensorAddress(engine->output_name.c_str(), engine->output_buffer.ptr)) {
    set_status(status, 24, "failed to bind scores tensor");
    return false;
  }
  if (!engine->context->enqueueV3(engine->stream)) {
    set_status(status, 25, "TensorRT enqueueV3 failed for reranker engine");
    return false;
  }
  if (!cuda_ok(cudaMemcpyAsync(output_scores, engine->output_buffer.ptr, output_bytes,
                               cudaMemcpyDeviceToHost, engine->stream),
               "failed to copy reranker scores from GPU", status) ||
      !cuda_ok(cudaStreamSynchronize(engine->stream), "cudaStreamSynchronize failed", status)) {
    return false;
  }
  return true;
}

}  // extern "C"
