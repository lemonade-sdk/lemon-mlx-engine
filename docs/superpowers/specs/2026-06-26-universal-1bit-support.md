# Universal 1-bit Model Support

## Goal
Every 1-bit model on HuggingFace loads and generates correctly in lemon-mlx-engine.

## Architecture Landscape

### 1-bit Model Types on HuggingFace (891 models total)

| Format | Architecture | Quant Approach | Examples | Engine Status |
|--------|-------------|----------------|----------|---------------|
| BitNet native | model_type=bitnet | U8 packed ternary + weight_scale | microsoft/bitnet-b1.58-2B-4T, tiiuae/Falcon-E-3B | ✅ BitNetModel |
| MLX 1-bit affine | model_type=qwen3/llama | U32 packed + .scales/.biases | prism-ml/Bonsai-*-mlx-1bit | ✅ Qwen3Model/LlamaModel |
| 1bitLLM offline | model_type=llama, weight_bits=1 | F32 ternary + weight_scale | 1bitLLM/bitnet_b1_58-3B | ✅ LlamaModel |
| HF quant_method=bitnet | model_type=llama/qwen3 | quant_method=bitnet + linear_class | tiiuae/Falcon3-7B, codys12/Qwen3-8B-BitNet | ✅ BitNetModel/Qwen3Model |
| AQLM 1-bit | model_type=llama | AQLM format (ISTA-DASLab) | ISTA-DASLab/Llama-*-AQLM-PV-1Bit | ⚠️ Needs AQLM support |
| PTQTP 1.58bit | model_type=qwen3/llama | PTQTP format (yang31210999) | yang31210999/Qwen3-*-PTQTP-1.58b | ⚠️ qwen3 works, llama needs testing |
| EdgeRazor 1.58bit | model_type=qwen3 | EdgeRazor | zhangsq-nju/Qwen3-*-EdgeRazor-1.58bit | ⚠️ Needs testing |
| MobileLLM ParetoQ | facebook/MobileLLM-ParetoQ-* | Custom ParetoQ | facebook/MobileLLM-ParetoQ-1.5B-1-bit | ❌ New architecture |
| GGUF 1-bit | Various | GGUF wrappers (TQ2_0, Q1_0) | prism-ml/Bonsai-*-gguf, mradermacher/*-GGUF | ⚠️ GGUF path exists |

## Implementation Plan

### Phase 1: Robust Detection (llm_factory.cpp)
Add comprehensive 1-bit detection that checks ALL possible config locations:

```cpp
// Check locations in priority order:
1. weight_bits == 1 (top-level)
2. input_bits == 8 (top-level)
3. quantization.bits == 1 (MLX format)
4. quantization_config.bits == 1 (HF format)
5. quantization_config.quant_method == "bitnet"
6. quantization.quant_method == "bitnet"
7. quantization_config.linear_class exists (implies BitLinear)
8. Check *.weight_scale in safetensors header (BitNet format)
```

### Phase 2: Universal Weight Prefix Stripping (llm_factory.cpp)
Add comprehensive prefix stripping BEFORE registration:

```cpp
prefix_strips = {
    {"language_model.model.", "model."},  // Gemma 4
    {"language_model.", ""},              // General multimodal
    {"model.model.", "model."},           // Nested
    {"llama.", "model."},                 // Legacy
    {"transformer.", "model."},           // GPT-style
    {"gpt_neox.", "model."},              // Neox-style
}
```

### Phase 3: AQLM Support
AQLM 1-bit is a unique format. Need to research and implement.

### Phase 4: Unified Test Framework
Script that:
1. Downloads model
2. Loads it in the engine
3. Runs 3-token generation
4. Reports success/failure with error message

### Phase 5: Registration
Add all verified models to the registry with example prompts.
