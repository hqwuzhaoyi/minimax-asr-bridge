# minimax-asr-bridge

A tiny OpenAI-compatible speech-to-text endpoint backed by **MiniMax `asr-1.0`**.

```
client ──POST /v1/audio/transcriptions (multipart)──▶ bridge
bridge ──POST /v1/speech_to_text (multipart, asr-1.0)──▶ api.minimaxi.com
bridge ◀──────────────────────── {"text": "..."} ────────┘
client ◀──────────────────────── {"text": "..."} ────────┘
```

Any client that speaks OpenAI's `/v1/audio/transcriptions` — the OpenAI SDKs,
[self-hosted memos](https://github.com/usememos/memos) voice transcription,
LiteLLM's `openai/` audio provider, home-automation glue — can now use MiniMax
ASR with zero code changes.

## Why

MiniMax's ASR is cheap and strong on Chinese, but it only exposes a private
API shape (`POST /v1/speech_to_text`). Existing gateways (LiteLLM, one-api,
new-api, Magpie, …) route `/v1/audio/transcriptions` to OpenAI-style STT
upstreams only, so MiniMax ASR is unreachable from OpenAI-SDK clients. This
bridge is the missing adapter — written in Rust + Tokio (axum), single static
binary, ~15 MB Docker image.

## Configuration

| Env                | Default                    | Meaning                                  |
| ------------------ | -------------------------- | ---------------------------------------- |
| `MINIMAX_API_KEY`  | (required)                 | MiniMax platform API key                 |
| `MINIMAX_API_BASE` | `https://api.minimaxi.com` | MiniMax API base (international host)    |
| `MINIMAX_MODEL`    | `asr-1.0`                  | Upstream ASR model                       |
| `BRIDGE_API_KEY`   | (unset = open)             | Require inbound `Authorization: Bearer`  |
| `LISTEN_ADDR`      | `0.0.0.0:8090`             | Listen address                           |

## Endpoints

- `POST /v1/audio/transcriptions` — OpenAI multipart format (`file` required;
  `model`, `language`, `prompt`, `response_format` accepted and ignored — the
  upstream model is fixed by `MINIMAX_MODEL`). Returns `{"text": "..."}`;
  `response_format`/`x-response-format: verbose_json` returns
  `{"task", "language", "duration", "text", "segments"}`.
- `POST /v1/audio/translations` — same handler, for SDKs that POST there.
- `GET /healthz` — liveness.

Upstream MiniMax errors (`base_resp.status_code != 0`) are surfaced as
OpenAI-style `{"error": {"message", "type"}}` with HTTP 502, so client retry /
circuit-breaker logic keeps working.

## Run

```bash
export MINIMAX_API_KEY=...
cargo run --release
# or
docker build -t minimax-asr-bridge . && docker run --rm -p 8090:8090 \
  -e MINIMAX_API_KEY=... minimax-asr-bridge
```

Point your client at `http://<host>:8090/v1` as an OpenAI-compatible provider
and use any model name (it is ignored upstream).

## Tests

```bash
cargo test        # unit + router tests incl. mock-upstream end-to-end
cargo clippy --all-targets
```

## License

MIT
