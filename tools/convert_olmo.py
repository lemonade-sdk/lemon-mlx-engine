#!/usr/bin/env python3
"""Convert OLMo-1 weights to standard Llama format for lemon-mlx-engine.

OLMo-1 uses fused QKV attention (att_proj = Q+K+V stacked) and fused
FFN gate+up (ff_proj = gate+up stacked). This script splits them into
the standard Llama format the engine expects.

Usage:
    python3 convert_olmo.py /path/to/olmo-model-directory
"""

import sys, os, json, struct
import torch
from safetensors.torch import save_file

def convert_olmo(model_dir: str):
    """Convert OLMo safetensors or pytorch .bin to Llama format."""
    # Find input file
    src = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(src):
        src = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(src):
        print(f"No model file found in {model_dir}")
        return False
    
    ext = os.path.splitext(src)[1]
    if ext == '.bin':
        sd = torch.load(src, map_location="cpu", mmap=True)
    else:
        from safetensors import safe_open
        sd = {}
        with safe_open(src, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    
    sd = {k: v for k, v in sd.items() if not k.startswith('optimizer')}
    
    output = {}
    for old_key, tensor in sd.items():
        # Parse OLMo key
        if old_key.startswith("model.transformer.blocks."):
            # model.transformer.blocks.{N}.{var}
            rest = old_key[25:]  # len("model.transformer.blocks.") = 25
            parts = rest.split('.')
            layer = parts[0]
            var = '.'.join(parts[1:])
            base = f"model.layers.{layer}."
        elif old_key.startswith("model.transformer."):
            var = old_key[18:]  # len("model.transformer.") = 18
            base = "model."
        else:
            output[old_key] = tensor
            continue
        
        # Remap
        if var == "att_proj.weight":
            h = tensor.shape[0] // 3
            output[base + "self_attn.q_proj.weight"] = tensor[:h]
            output[base + "self_attn.k_proj.weight"] = tensor[h:2*h]
            output[base + "self_attn.v_proj.weight"] = tensor[2*h:]
        elif var == "ff_proj.weight":
            h = tensor.shape[0] // 2
            output[base + "mlp.gate_proj.weight"] = tensor[:h]
            output[base + "mlp.up_proj.weight"] = tensor[h:]
        elif var == "attn_out.weight":
            output[base + "self_attn.o_proj.weight"] = tensor
        elif var == "attn_norm.weight":
            output[base + "input_layernorm.weight"] = tensor
        elif var == "ff_norm.weight":
            output[base + "post_attention_layernorm.weight"] = tensor
        elif var == "ff_out.weight":
            output[base + "mlp.down_proj.weight"] = tensor
        elif var == "ln_f.weight":
            output["model.norm.weight"] = tensor
        elif var == "wte.weight":
            output["model.embed_tokens.weight"] = tensor
    
    out_path = os.path.join(model_dir, "model.safetensors")
    save_file(output, out_path)
    print(f"✅ Converted {len(output)} tensors → {out_path}")
    print(f"   Size: {os.path.getsize(out_path)/1e6:.1f} MB")
    print(f"   Keys: {len([k for k in output if 'q_proj' in k])} q_proj, "
          f"{len([k for k in output if 'k_proj' in k])} k_proj, "
          f"{len([k for k in output if 'v_proj' in k])} v_proj")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_olmo.py /path/to/olmo-model-directory")
        sys.exit(1)
    convert_olmo(sys.argv[1])
