# Universal Hugging Face Model Loading Path

## Problem
`lemon-mlx-engine` only loads MLX-format HF repos (`mlx-community/*`). Arbitrary HF repos fail because:
1. Download hardcodes `config.json`, `tokenizer.json`, `model.safetensors` filenames
2. No `tokenizer.model` (SentencePiece) fallback
3. No `.safetensors` glob for non-standard shard names
4. Silent zero-fill on missing weight keys
5. Cryptic `Unsupported model type` error
6. No `quantization_config` reading from `config.json`

## Design

### Phase 1: Universal download (`src/common/hub_api.cpp`)
Replace `snapshot_download`'s hardcoded file list with HF API file enumeration:
- `GET https://huggingface.co/api/models/{repo_id}/revision/{rev}` returns `{siblings: [{rfilename: "..."}]}`
- Download every file matching: `*.json`, `*.safetensors`, `*.token`, `*.model`, `*.txt`, `*.jinja`
- Skip `*.bin`, `*.pt`, `*.h5`, `*.msgpack` (PyTorch/native formats — too large to load without conversion)
- Preserve existing cache-check shortcut (`config.json` exists → return)

### Phase 2: Universal tokenizer (`src/common/tokenizer.cpp`, `include/.../tokenizer.h`)
- Try `tokenizer.json` first (current behavior)
- If missing, try `tokenizer.model` via `tokenizers_cpp::Tokenizer::FromBlobSentencePiece()`
- If missing, try `vocab.json` + `merges.txt` via `Tokenizer::FromBlobJSON` reconstruction
- Throw clear error listing what was tried

### Phase 3: Weight loading robustness (`src/common/safetensors.cpp`, `src/llm/llm_factory.cpp`)
- `load_weights`: count found vs missing keys; `cerr` WARNING if any missing
- Unknown `model_type`: list all 52 supported types in the error
- Read `quantization_config` from `config.json` in `parse_base_configuration`

### Phase 4: Model-type aliases (`src/llm/llm_factory.cpp`)
- Add alias map: `{mistral→llama, acereason→qwen2, command-r→cohere, phi3small→phi3, ...}`
- Before failing on unknown `model_type`, check aliases

## Out of scope
- GGUF loading (needs libllama C++ integration)
- PyTorch `.bin`/`.pt` checkpoint conversion (needs torch dependency)
- On-the-fly quantization of unquantized models (separate feature)
- `trust_remote_code` dynamic model loading (C++ can't exec Python)

## Testing
- Unit test: `snapshot_download` enumerates via API (mock or real small repo)
- Unit test: tokenizer fallback to SentencePiece
- Unit test: missing-weight warning triggers
- Integration: download + load `mlx-community/Falcon-E-3B-Instruct-1.58-bit` from repo ID
- Integration: verify BitNet-2B, Llama-1B, Falcon-E still work after changes