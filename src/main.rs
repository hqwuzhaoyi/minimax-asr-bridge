//! minimax-asr-bridge — an OpenAI-compatible `/v1/audio/transcriptions`
//! endpoint backed by MiniMax `asr-1.0` (`/v1/speech_to_text`).
//!
//! Lets any OpenAI-SDK-speaking client (e.g. self-hosted memos voice
//! transcription) use MiniMax ASR without code changes:
//!
//! ```text
//! client ──POST /v1/audio/transcriptions (multipart)──▶ bridge
//! bridge ──POST /v1/speech_to_text (multipart, asr-1.0)──▶ api.minimaxi.com
//! bridge ◀────────────────────── {"text": "..."} ─────────┘
//! client ◀────────────────────── {"text": "..."} ─────────┘
//! ```

use std::net::SocketAddr;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

/// Max upload accepted from clients (OpenAI caps transcriptions at 25 MB).
const MAX_BODY_BYTES: usize = 26 * 1024 * 1024;

#[derive(Clone)]
struct AppState {
    http: reqwest::Client,
    upstream_base: String,
    upstream_model: String,
    minimax_key: String,
    /// Optional inbound bearer; when set, clients must present it.
    bridge_key: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let minimax_key = std::env::var("MINIMAX_API_KEY")
        .map_err(|_| anyhow::anyhow!("MINIMAX_API_KEY is required (MiniMax platform API key)"))?;
    let upstream_base = std::env::var("MINIMAX_API_BASE")
        .unwrap_or_else(|_| "https://api.minimaxi.com".to_string())
        .trim_end_matches('/')
        .to_string();
    let upstream_model = std::env::var("MINIMAX_MODEL").unwrap_or_else(|_| "asr-1.0".to_string());
    let bridge_key = std::env::var("BRIDGE_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let listen: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8090".to_string())
        .parse()?;

    let state = AppState {
        http: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?,
        upstream_base,
        upstream_model,
        minimax_key,
        bridge_key,
    };

    let app = router(state);
    tracing::info!(%listen, "listening");
    let listener = tokio::net::TcpListener::bind(listen).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/translations", post(transcribe))
        .route("/healthz", get(health))
        .route("/", get(index))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}

async fn health() -> &'static str {
    "ok"
}

async fn index() -> Json<Value> {
    Json(json!({
        "service": "minimax-asr-bridge",
        "endpoints": ["/v1/audio/transcriptions", "/v1/audio/translations"],
        "upstream_model_env": "MINIMAX_MODEL (default asr-1.0)",
    }))
}

#[derive(Debug)]
struct AudioUpload {
    filename: String,
    content_type: Option<String>,
    data: Bytes,
}

/// Extract the `file` part; other OpenAI fields (`model`, `language`,
/// `prompt`, `response_format`, …) are accepted and ignored — the upstream
/// model is fixed by `MINIMAX_MODEL`.
#[allow(clippy::result_large_err)]
async fn read_upload(mut multipart: Multipart) -> Result<AudioUpload, Response> {
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        openai_error(
            StatusCode::BAD_REQUEST,
            format!("invalid multipart body: {e}"),
        )
    })? {
        if field.name() == Some("file") {
            let filename = field.file_name().unwrap_or("audio").to_string();
            let content_type = field.content_type().map(str::to_string);
            let data = field.bytes().await.map_err(|e| {
                openai_error(StatusCode::BAD_REQUEST, format!("failed to read file: {e}"))
            })?;
            if data.is_empty() {
                return Err(openai_error(StatusCode::BAD_REQUEST, "file part is empty"));
            }
            return Ok(AudioUpload {
                filename,
                content_type,
                data,
            });
        }
    }
    Err(openai_error(
        StatusCode::BAD_REQUEST,
        "multipart field 'file' is required (as OpenAI's /v1/audio/transcriptions)",
    ))
}

async fn transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Response {
    if let Err(resp) = check_inbound_auth(&state, &headers) {
        return resp;
    }

    let upload = match read_upload(multipart).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };
    tracing::info!(
        filename = %upload.filename,
        content_type = upload.content_type.as_deref().unwrap_or("unknown"),
        bytes = upload.data.len(),
        "transcription request"
    );

    let part = reqwest::multipart::Part::bytes(upload.data.to_vec())
        .file_name(upload.filename)
        .mime_str(
            upload
                .content_type
                .as_deref()
                .unwrap_or("application/octet-stream"),
        )
        .expect("static mime string");
    let form = reqwest::multipart::Form::new()
        .text("model", state.upstream_model.clone())
        .part("file", part);

    let url = format!("{}/v1/speech_to_text", state.upstream_base);
    let resp = match state
        .http
        .post(&url)
        .bearer_auth(&state.minimax_key)
        .multipart(form)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "upstream request failed");
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {e}"),
            );
        }
    };

    let status = resp.status();
    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream read failed: {e}"),
            );
        }
    };

    match map_upstream_body(status.as_u16(), &body, wants_verbose(&headers)) {
        Ok(value) => (StatusCode::OK, Json(value)).into_response(),
        Err(err) => {
            tracing::warn!(upstream_status = status.as_u16(), error = %err, "upstream error");
            openai_error(StatusCode::BAD_GATEWAY, err)
        }
    }
}

#[allow(clippy::result_large_err)]
fn check_inbound_auth(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = state.bridge_key.as_deref() else {
        return Ok(());
    };
    let got = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match got {
        Some(k) if constant_time_eq(k.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err(openai_error(
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key",
        )),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn wants_verbose(headers: &HeaderMap) -> bool {
    // response_format is normally a form field, but clients that pre-read it
    // into a header are honoured too; the multipart field is parsed upstream
    // and mapped the same way.
    headers
        .get("x-response-format")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "verbose_json")
        .unwrap_or(false)
}

/// Map MiniMax `/v1/speech_to_text` response into an OpenAI-style response.
///
/// MiniMax success: `{"text": "...", "duration": 2.8, "trace_id": "..."}`
/// MiniMax failure: `{"base_resp": {"status_code": 1004, "status_msg": "..."}}`
fn map_upstream_body(status: u16, body: &str, verbose: bool) -> Result<Value, String> {
    let parsed: Value = serde_json::from_str(body)
        .map_err(|e| format!("upstream returned non-JSON body (HTTP {status}): {e}"))?;

    if let Some(base) = parsed.get("base_resp") {
        let code = base.get("status_code").and_then(Value::as_i64).unwrap_or(0);
        if code != 0 {
            let msg = base
                .get("status_msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown upstream error");
            return Err(format!("minimax error {code}: {msg}"));
        }
    }

    let text = parsed
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("upstream response has no 'text' field (HTTP {status})"))?;

    if verbose {
        let duration = parsed
            .get("duration")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        Ok(json!({
            "task": "transcribe",
            "language": null,
            "duration": duration,
            "text": text,
            "segments": [],
        }))
    } else {
        Ok(json!({ "text": text }))
    }
}

fn openai_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": { "message": message.into(), "type": "invalid_request_error" }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn maps_success_to_openai_shape() {
        let body = r#"{"text":"你好，世界","duration":2.8,"trace_id":"t"}"#;
        let v = map_upstream_body(200, body, false).unwrap();
        assert_eq!(v["text"], "你好，世界");
        assert!(v.get("duration").is_none());
    }

    #[tokio::test]
    async fn maps_verbose_shape() {
        let body = r#"{"text":"hi","duration":1.5}"#;
        let v = map_upstream_body(200, body, true).unwrap();
        assert_eq!(v["task"], "transcribe");
        assert_eq!(v["duration"], 1.5);
    }

    #[tokio::test]
    async fn surfaces_base_resp_errors() {
        let body = r#"{"base_resp":{"status_code":1004,"status_msg":"invalid api key"}}"#;
        let err = map_upstream_body(200, body, false).unwrap_err();
        assert!(err.contains("minimax error 1004"), "got: {err}");
    }

    #[tokio::test]
    async fn rejects_body_without_text() {
        let err = map_upstream_body(200, r#"{"foo":1}"#, false).unwrap_err();
        assert!(err.contains("no 'text' field"));
    }

    #[tokio::test]
    async fn missing_file_part_is_400() {
        let state = test_state();
        let app = router(state);
        // multipart without a "file" part
        let body =
            "--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--B--\r\n";
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer bridge-key")
            .header("content-type", "multipart/form-data; boundary=B")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("'file' is required")
        );
    }

    #[tokio::test]
    async fn end_to_end_with_mock_upstream() {
        // stub MiniMax upstream
        let upstream = Router::new().route(
            "/v1/speech_to_text",
            post(|headers: HeaderMap, mut mp: Multipart| async move {
                assert_eq!(
                    headers.get("authorization").unwrap(),
                    "Bearer minimax-secret"
                );
                let mut got_model = None;
                let mut got_audio: Option<usize> = None;
                while let Some(f) = mp.next_field().await.unwrap() {
                    match f.name() {
                        Some("model") => got_model = Some(f.text().await.unwrap()),
                        Some("file") => {
                            got_audio = Some(f.bytes().await.unwrap().len());
                        }
                        _ => {}
                    }
                }
                assert_eq!(got_model.as_deref(), Some("asr-1.0"));
                assert!(got_audio.unwrap_or(0) > 0);
                Json(json!({"text":"mock 转写结果","duration":1.0}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let state = AppState {
            http: reqwest::Client::new(),
            upstream_base: format!("http://{addr}"),
            upstream_model: "asr-1.0".into(),
            minimax_key: "minimax-secret".into(),
            bridge_key: Some("bridge-key".into()),
        };
        let app = router(state);

        let body = "--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFxxxx\r\n--B--\r\n";
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer bridge-key")
            .header("content-type", "multipart/form-data; boundary=B")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["text"], "mock 转写结果");
    }

    #[tokio::test]
    async fn rejects_bad_inbound_key() {
        let state = test_state();
        let app = router(state);
        let body = "--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nx\r\n--B--\r\n";
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer wrong")
            .header("content-type", "multipart/form-data; boundary=B")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    fn test_state() -> AppState {
        AppState {
            http: reqwest::Client::new(),
            upstream_base: "http://127.0.0.1:1".into(),
            upstream_model: "asr-1.0".into(),
            minimax_key: "k".into(),
            bridge_key: Some("bridge-key".into()),
        }
    }
}
