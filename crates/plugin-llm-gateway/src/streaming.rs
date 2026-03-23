use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use hyper::body::{Body, Frame};
use tokio::sync::oneshot;

/// Usage info extracted from the tail of an SSE stream.
#[derive(Debug, Clone)]
pub struct StreamUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Body wrapper that passes all frames through immediately while keeping
/// a rolling tail buffer. On stream end, it parses the tail for usage info
/// and sends the result through a oneshot channel.
pub struct UsageExtractorBody {
    inner: BoxBody<Bytes, hyper::Error>,
    tail_buf: Vec<u8>,
    tx: Option<oneshot::Sender<Option<StreamUsage>>>,
}

/// Maximum bytes to keep in the tail buffer for usage extraction.
const TAIL_CAPACITY: usize = 2048;

impl UsageExtractorBody {
    pub fn new(
        inner: BoxBody<Bytes, hyper::Error>,
    ) -> (Self, oneshot::Receiver<Option<StreamUsage>>) {
        let (tx, rx) = oneshot::channel();
        let body = Self {
            inner,
            tail_buf: Vec::with_capacity(TAIL_CAPACITY),
            tx: Some(tx),
        };
        (body, rx)
    }

    fn append_to_tail(&mut self, data: &[u8]) {
        self.tail_buf.extend_from_slice(data);
        if self.tail_buf.len() > TAIL_CAPACITY {
            let excess = self.tail_buf.len() - TAIL_CAPACITY;
            self.tail_buf.drain(..excess);
        }
    }

    fn extract_and_send(&mut self) {
        let usage = extract_usage_from_tail(&self.tail_buf);
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(usage);
        }
    }
}

impl Body for UsageExtractorBody {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();

        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.append_to_tail(data);
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(e))) => {
                // Stream error — send None and pass error through
                this.extract_and_send();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                // Stream ended — extract usage from tail
                this.extract_and_send();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.inner.size_hint()
    }
}

/// Extract token usage from the tail buffer by scanning for OpenAI or Anthropic
/// usage JSON patterns.
///
/// OpenAI format: `"usage":{"prompt_tokens":N,"completion_tokens":N,"total_tokens":N}`
/// Anthropic format: `"usage":{"input_tokens":N,"output_tokens":N}`
fn extract_usage_from_tail(tail: &[u8]) -> Option<StreamUsage> {
    let text = std::str::from_utf8(tail).ok()?;

    // Find last occurrence of "usage" to handle multiple chunks
    let usage_idx = text.rfind("\"usage\"")?;
    let after_usage = &text[usage_idx..];

    // Find the opening brace after "usage"
    let brace_start = after_usage.find('{')?;
    let usage_obj = &after_usage[brace_start..];

    // Find matching close brace (simple: find first '}')
    let brace_end = usage_obj.find('}')?;
    let usage_str = &usage_obj[..=brace_end];

    // Extract fields
    let prompt_tokens = extract_int_field(usage_str, "prompt_tokens")
        .or_else(|| extract_int_field(usage_str, "input_tokens"));
    let completion_tokens = extract_int_field(usage_str, "completion_tokens")
        .or_else(|| extract_int_field(usage_str, "output_tokens"));
    let total_tokens = extract_int_field(usage_str, "total_tokens");

    if prompt_tokens.is_some() || completion_tokens.is_some() {
        Some(StreamUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        })
    } else {
        None
    }
}

/// Extract an integer value for a JSON key like `"field_name":123`.
fn extract_int_field(text: &str, field: &str) -> Option<u64> {
    let pattern = format!("\"{}\"", field);
    let idx = text.find(&pattern)?;
    let after = &text[idx + pattern.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();

    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    if end == 0 {
        return None;
    }
    after[..end].parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use http_body_util::BodyExt;

    #[test]
    fn test_extract_openai_usage() {
        let tail = br#"data: {"id":"chatcmpl-123","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}

data: [DONE]"#;

        let usage = extract_usage_from_tail(tail).unwrap();
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(30));
    }

    #[test]
    fn test_extract_anthropic_usage() {
        let tail = br#"event: message_delta
data: {"type":"message_delta","usage":{"input_tokens":15,"output_tokens":25}}

event: message_stop"#;

        let usage = extract_usage_from_tail(tail).unwrap();
        assert_eq!(usage.prompt_tokens, Some(15));
        assert_eq!(usage.completion_tokens, Some(25));
    }

    #[test]
    fn test_no_usage_returns_none() {
        let tail = br#"data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"hello"}}]}

data: [DONE]"#;

        assert!(extract_usage_from_tail(tail).is_none());
    }

    #[test]
    fn test_tail_buffer_truncation() {
        let mut body_wrapper = UsageExtractorBody {
            inner: http_body_util::Empty::new()
                .map_err(|never| -> hyper::Error { match never {} })
                .boxed(),
            tail_buf: Vec::new(),
            tx: None,
        };

        // Append more than TAIL_CAPACITY bytes
        let large_data = vec![b'x'; TAIL_CAPACITY + 500];
        body_wrapper.append_to_tail(&large_data);

        assert_eq!(body_wrapper.tail_buf.len(), TAIL_CAPACITY);
    }

    #[tokio::test]
    async fn test_body_passthrough() {
        use http_body_util::Full;

        let original_data = Bytes::from("hello world stream data");
        let inner = Full::new(original_data.clone())
            .map_err(|never| -> hyper::Error { match never {} })
            .boxed();

        let (wrapper, rx) = UsageExtractorBody::new(inner);
        let boxed: BoxBody<Bytes, hyper::Error> = BoxBody::new(wrapper);

        let collected = boxed.collect().await.unwrap().to_bytes();
        assert_eq!(collected, original_data);

        // No usage in this data
        let usage = rx.await.unwrap();
        assert!(usage.is_none());
    }

    #[tokio::test]
    async fn test_body_extracts_usage_from_stream() {
        use http_body_util::Full;

        let sse_data = Bytes::from(
            "data: {\"choices\":[]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"total_tokens\":150}}\n\ndata: [DONE]\n\n"
        );
        let inner = Full::new(sse_data)
            .map_err(|never| -> hyper::Error { match never {} })
            .boxed();

        let (wrapper, rx) = UsageExtractorBody::new(inner);
        let boxed: BoxBody<Bytes, hyper::Error> = BoxBody::new(wrapper);

        // Consume the body
        let _ = boxed.collect().await.unwrap();

        let usage = rx.await.unwrap().unwrap();
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(50));
        assert_eq!(usage.total_tokens, Some(150));
    }

    #[test]
    fn test_extract_int_field() {
        assert_eq!(
            extract_int_field(
                r#"{"prompt_tokens":42,"completion_tokens":7}"#,
                "prompt_tokens"
            ),
            Some(42)
        );
        assert_eq!(
            extract_int_field(
                r#"{"prompt_tokens":42,"completion_tokens":7}"#,
                "completion_tokens"
            ),
            Some(7)
        );
        assert_eq!(extract_int_field(r#"{"foo":"bar"}"#, "prompt_tokens"), None);
    }
}
