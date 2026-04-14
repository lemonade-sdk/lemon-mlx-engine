# Lemon MLX Engine

C++ inference engine for large language models, built on [MLX](https://github.com/ml-explore/mlx).

Run LLMs locally on **Apple M-series**, **AMD GPUs** (Linux/Windows), and CPU -- no Python required.

## Features

- **50+ LLM architectures** -- Llama, Qwen, Gemma, Phi, DeepSeek, Mistral, Granite, GLM, Falcon, and more
- **12 VLM architectures** -- Qwen-VL, PaliGemma, Pixtral, Gemma3, SmolVLM, and more
- **Embedders** -- BERT, Nomic-BERT, Qwen3-Embed
- **Quantized inference** -- 4-bit/8-bit via `quantized_matmul`
- **HuggingFace integration** -- auto-downloads models, tokenizers, and chat templates
- **OpenAI-compatible API server** -- drop-in replacement for local inference
- **Streaming generation** -- async token pipeline with KV caching
- **Multi-model management** -- LRU eviction, explicit load/unload
- **Chat templates** -- Jinja2-compatible (minja), auto-loaded from model config

## Requirements

- CMake 3.20+
- C++17 compiler
- libcurl
- Rust toolchain (for tokenizers-cpp)
- ROCm (for AMD GPU builds)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

For AMD GPU (ROCm):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_ROCM=ON
make -j
```

## Interactive Chat

```
./chat <model_id_or_path> [options]
```

Models are specified as HuggingFace repo IDs (auto-downloaded on first use) or local directory paths.

```bash
# Basic usage -- downloads the model on first run
./chat mlx-community/Qwen3-1.7B-4bit

# With system prompt and tuned sampling
./chat mlx-community/Qwen3-4B-4bit \
  --system-prompt "You are a helpful coding assistant" \
  --temperature 0.5 --max-tokens 4096

# Use a local model directory
./chat /path/to/my-model

# Disable thinking/reasoning for Qwen3 models
./chat mlx-community/Qwen3-8B-4bit --no-think

# KV cache quantization to save memory
./chat mlx-community/Qwen3-8B-4bit --kv-bits 4

# Raw mode (skip chat template)
./chat mlx-community/starcoder2-3b-4bit --raw
```

Type your message at the `>` prompt. Type `quit` or `exit` to leave.

### Chat Options

| Flag | Default | Description |
|------|---------|-------------|
| `--system-prompt "..."` | *(none)* | System instructions for the session |
| `--max-tokens N` | 2048 | Maximum tokens to generate per response |
| `--temperature T` | 0.7 | Sampling temperature (lower = more deterministic) |
| `--top-p P` | 0.9 | Nucleus sampling threshold |
| `--repetition-penalty F` | 0.0 (off) | Penalize token repetition |
| `--memory-limit MB` | 0 (unlimited) | GPU wired memory limit in MB |
| `--no-think` | false | Disable thinking/reasoning (Qwen3 models) |
| `--raw` | false | Skip chat template, use raw token encoding |
| `--kv-bits N` | 0 (off) | KV cache quantization bits (4 or 8) |
| `--kv-group-size N` | 64 | KV cache quantization group size |
| `--ctx-size N` | 0 (auto) | Pre-allocate KV cache for N tokens |

## API Server

```
./server [model_id_or_path] [options]
```

The server exposes an **OpenAI-compatible** HTTP API. It works in two modes:

- **Pre-load mode** -- load a specific model at startup
- **Auto-load mode** -- start empty, load models on demand from API requests

```bash
# Pre-load a model
./server mlx-community/Qwen3-4B-4bit

# Auto-load mode (no model pre-loaded)
./server

# Custom host/port with multiple model slots
./server --host 0.0.0.0 --port 9090 --max-loaded 3

# Offline mode (no HuggingFace downloads)
./server --no-download
```

### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host HOST` | 127.0.0.1 | Bind address |
| `--port PORT` | 8080 | Listen port |
| `--max-tokens N` | 2048 | Default max tokens for generation |
| `--temperature T` | 0.7 | Default sampling temperature |
| `--top-p P` | 0.9 | Default nucleus sampling threshold |
| `--repetition-penalty F` | 0.0 (off) | Default repetition penalty |
| `--memory-limit MB` | 0 (unlimited) | GPU wired memory limit |
| `--no-think` | false | Disable thinking/reasoning globally |
| `--no-download` | false | Don't auto-download models from HuggingFace |
| `--max-loaded N` | 1 | Max models in memory (LRU eviction) |
| `--kv-bits N` | 0 (off) | KV cache quantization bits (4 or 8) |
| `--kv-group-size N` | 64 | KV cache quantization group size |
| `--ctx-size N` | 0 (auto) | Pre-allocate KV cache for N tokens |

### API Endpoints

#### `GET /health`

Health check.

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

#### `GET /v1/models`

List available models. Returns all MLX models found in the HuggingFace cache, with loaded status.

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen3-4B-4bit",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local (loaded)"
    }
  ]
}
```

#### `POST /v1/chat/completions`

Chat completion (OpenAI-compatible). The model is auto-loaded if not already in memory.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-4bit",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

Request fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | *required* | Model ID (auto-loaded if needed) |
| `messages` | array | *required* | Chat history (`role` + `content`) |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling |
| `max_tokens` | int | 2048 | Max tokens to generate |
| `repetition_penalty` | float | 0.0 | Repetition penalty |
| `stream` | bool | false | Stream response via SSE |
| `stop` | array | [] | Stop sequences |

Streaming example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

#### `POST /v1/completions`

Text completion (non-chat). Same fields as chat completions but uses `prompt` instead of `messages`.

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-4bit",
    "prompt": "The answer to life is",
    "max_tokens": 100
  }'
```

#### `POST /load`

Explicitly load a model into memory.

```bash
curl -X POST http://localhost:8080/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3-8B-4bit"}'
```

#### `POST /unload`

Unload a model from memory.

```bash
curl -X POST http://localhost:8080/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3-8B-4bit"}'
```

### Using with OpenAI-compatible clients

The server works with any client that speaks the OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="mlx-community/Qwen3-4B-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Finding and Loading Models

Models are resolved in this order:

1. **Local directory** -- if the path contains `config.json`, use it directly
2. **HuggingFace cache** -- check for a previously downloaded snapshot
3. **HuggingFace Hub** -- download from the hub (unless `--no-download`)

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_HUB_CACHE` | Override HuggingFace cache directory (highest priority) |
| `HF_HOME` | HuggingFace home directory (uses `$HF_HOME/hub` as cache) |
| `HF_TOKEN` | API token for accessing private/gated models |

### Listing cached models

Use the `/v1/models` endpoint to see all MLX-compatible models in your HuggingFace cache. A model needs `config.json` and `.safetensors` files to be detected.

When running the server in auto-load mode, cached models are also printed to stderr at startup.

## Diagnose Tool

Pipeline diagnostic tool that validates each inference stage for numerical issues.

```bash
./diagnose mlx-community/Qwen3-4B-4bit
```

Tests GPU operations, quantized matmul, RMS normalization, RoPE, forward pass, dequantization, and end-to-end token generation. Reports NaN/Inf counts and numerical statistics for each stage.

## Supported Architectures

### LLM (Text)

Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE, Qwen3-Next, Qwen3.5-MoE, Gemma, Gemma2, Gemma3, Gemma3n, Phi, Phi3, PhiMoE, DeepSeek-V3, MiMo, Cohere/Command-R, Starcoder2, Mistral3, Granite, GraniteMoE-Hybrid, GLM4, GLM4-MoE, GLM4-MoE-Lite, Ernie4.5, SmolLM3, MiniCPM, OLMo2, OLMo3, OLMoE, NanoChat, Lille-130m, InternLM2, Exaone4, Apertus, OpenELM, BailingMoE, AFMoE, GPT-OSS, LFM2, LFM2-MoE, Baichuan-M1, Falcon-H1, Nemotron-H, Jamba, AceReason

### VLM (Vision + Language)

Qwen2-VL, Qwen2.5-VL, Qwen3-VL, PaliGemma, Gemma3, Pixtral, Mistral3, Idefics3, SmolVLM, FastVLM, LLaVA-Qwen2, LFM2-VL

### Embedders

BERT, Nomic-BERT, Qwen3-Embed

## Libraries

| Library | Description |
|---------|-------------|
| `mlx-lm-core` | MLX module wrappers |
| `mlx-lm-common` | Tokenizer, generation, KV cache, hub API |
| `mlx-lm-llm` | LLM model implementations |
| `mlx-lm-vlm` | Vision-language model implementations |
| `mlx-lm-embedders` | Embedding model implementations |

## License

MIT
