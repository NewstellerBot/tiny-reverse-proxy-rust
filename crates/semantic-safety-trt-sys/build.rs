use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TENSORRT_HEADER: &str = "NvInfer.h";
const CUDA_HEADER: &str = "cuda_runtime_api.h";
const TENSORRT_LIB_PREFIX: &str = "libnvinfer.so";
const CUDA_LIB_PREFIX: &str = "libcudart.so";

fn env_dir(name: &str) -> Option<PathBuf> {
    let value = env::var(name).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        panic!("{name} must not be empty when native-tensorrt is enabled");
    }
    let path = PathBuf::from(trimmed);
    if !path.is_dir() {
        panic!(
            "{name} must point to an existing directory when native-tensorrt is enabled: {}",
            path.display()
        );
    }
    Some(path)
}

fn contains_file(dir: &Path, name: &str) -> bool {
    dir.join(name).is_file()
}

fn contains_library_prefix(dir: &Path, prefix: &str) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .file_name()
            .to_str()
            .map(|name| name == prefix || name.starts_with(&format!("{prefix}.")))
            .unwrap_or(false)
    })
}

fn resolve_header_dir(env_name: &str, header: &str, fallbacks: &[&str]) -> PathBuf {
    if let Some(dir) = env_dir(env_name) {
        if contains_file(&dir, header) {
            return dir;
        }
        panic!(
            "{env_name} does not contain {header}: {}",
            dir.join(header).display()
        );
    }

    for candidate in fallbacks {
        let path = Path::new(candidate);
        if contains_file(path, header) {
            return path.to_path_buf();
        }
    }

    panic!(
        "could not find {header}; set {env_name} to the directory containing it before building with --features native-tensorrt"
    );
}

fn resolve_library_dir(env_name: &str, library_prefix: &str, fallbacks: &[&str]) -> PathBuf {
    if let Some(dir) = env_dir(env_name) {
        if contains_library_prefix(&dir, library_prefix) {
            return dir;
        }
        panic!(
            "{env_name} does not contain a library matching {library_prefix}: {}",
            dir.display()
        );
    }

    for candidate in fallbacks {
        let path = Path::new(candidate);
        if contains_library_prefix(path, library_prefix) {
            return path.to_path_buf();
        }
    }

    panic!(
        "could not find a library matching {library_prefix}; set {env_name} to the directory containing it before building with --features native-tensorrt"
    );
}

fn main() {
    println!("cargo:rerun-if-env-changed=SEMANTIC_SAFETY_TENSORRT_INCLUDE");
    println!("cargo:rerun-if-env-changed=SEMANTIC_SAFETY_TENSORRT_LIB");
    println!("cargo:rerun-if-env-changed=SEMANTIC_SAFETY_CUDA_INCLUDE");
    println!("cargo:rerun-if-env-changed=SEMANTIC_SAFETY_CUDA_LIB");
    println!("cargo:rerun-if-changed=src/bridge.cc");

    if env::var_os("CARGO_FEATURE_NATIVE_TENSORRT").is_none() {
        return;
    }

    let tensorrt_include = resolve_header_dir(
        "SEMANTIC_SAFETY_TENSORRT_INCLUDE",
        TENSORRT_HEADER,
        &[
            "/usr/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/local/include",
            "/usr/local/tensorrt/include",
            "/opt/tensorrt/include",
        ],
    );
    let cuda_include = resolve_header_dir(
        "SEMANTIC_SAFETY_CUDA_INCLUDE",
        CUDA_HEADER,
        &[
            "/usr/local/cuda/include",
            "/usr/include",
            "/usr/include/x86_64-linux-gnu",
        ],
    );
    let tensorrt_lib = resolve_library_dir(
        "SEMANTIC_SAFETY_TENSORRT_LIB",
        TENSORRT_LIB_PREFIX,
        &[
            "/usr/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib",
            "/usr/local/tensorrt/lib",
            "/usr/local/tensorrt/lib64",
            "/opt/tensorrt/lib",
            "/opt/tensorrt/lib64",
        ],
    );
    let cuda_lib = resolve_library_dir(
        "SEMANTIC_SAFETY_CUDA_LIB",
        CUDA_LIB_PREFIX,
        &[
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ],
    );

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .include(&tensorrt_include)
        .include(&cuda_include)
        .file("src/bridge.cc");
    build.compile("semantic_safety_trt_bridge");

    println!("cargo:rustc-link-search=native={}", tensorrt_lib.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
