// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/safetensors.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlx_lm {

std::unordered_map<std::string, mlx::core::array>
load_safetensors(const std::string& path) {
    // MLX C++ has built-in safetensors loading
    return mlx::core::load_safetensors(path).first;
}

std::unordered_map<std::string, mlx::core::array>
load_safetensors_from_directory(const std::string& directory) {
    std::unordered_map<std::string, mlx::core::array> all_weights;

    // Check for sharded model FIRST (preferred over single file, because
    // HF Hub sometimes leaves a stub "model.safetensors" alongside shards)
    auto index_path = fs::path(directory) / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        std::ifstream f(index_path);
        nlohmann::json index;
        f >> index;

        if (index.contains("weight_map")) {
            std::set<std::string> shard_files;
            for (auto& [key, val] : index["weight_map"].items()) {
                shard_files.insert(val.get<std::string>());
            }

            for (const auto& shard_name : shard_files) {
                auto shard_path = fs::path(directory) / shard_name;
                if (!fs::exists(shard_path)) {
                    throw std::runtime_error(
                        "Missing shard file: " + shard_path.string());
                }
                auto shard_weights = load_safetensors(shard_path.string());
                for (auto& [k, v] : shard_weights) {
                    all_weights.insert_or_assign(k, std::move(v));
                }
            }
            return all_weights;
        }
    }

    // Check for single model.safetensors (must be large enough to be real)
    auto single_path = fs::path(directory) / "model.safetensors";
    if (fs::exists(single_path) && fs::file_size(single_path) > 1024) {
        return load_safetensors(single_path.string());
    }

    // Fallback: load all .safetensors files in the directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".safetensors") {
            auto weights = load_safetensors(entry.path().string());
            for (auto& [k, v] : weights) {
                all_weights.insert_or_assign(k, std::move(v));
            }
        }
    }

    if (all_weights.empty()) {
        // No safetensors found. Try PyTorch .bin files.
        // Write a temp Python conversion script and execute it.
        auto bin_path = fs::path(directory) / "pytorch_model.bin";
        if (!fs::exists(bin_path)) {
            // Try sharded pytorch format
            auto index_path = fs::path(directory) / "pytorch_model.bin.index.json";
            if (fs::exists(index_path)) {
                // Sharded .bin files — convert each shard
                bin_path = fs::path(directory);
            } else {
                throw std::runtime_error(
                    "No .safetensors files found in " + directory +
                    ". Install safetensors: pip install safetensors");
            }
        }

        std::cerr << "[convert] No safetensors found, attempting PyTorch .bin conversion...\n";

        // Write a conversion Python script
        std::string script_path = (fs::temp_directory_path() / "_mlx_convert_bin.py").string();
        std::string out_path = (fs::path(directory) / "model.safetensors").string();

        std::ofstream out(script_path);
        out << R"PY(
import json, os, sys

# Determine input: single .bin file or sharded index
input_dir = sys.argv[1]
single_bin = os.path.join(input_dir, "pytorch_model.bin")
sharded_index = os.path.join(input_dir, "pytorch_model.bin.index.json")
out_dir = sys.argv[1]

try:
    from safetensors.torch import save_file as st_save
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "safetensors", "-q", "--quiet"], check=True)
    from safetensors.torch import save_file as st_save

try:
    import torch
except ImportError:
    print("torch not available, trying to load from file...")
    # Some .bin files are just pickle dictionaries without requiring torch
    import pickle
    torch_load = lambda f: pickle.load(open(f, "rb"), encoding="bytes")
else:
    torch_load = lambda f: torch.load(f, map_location="cpu", weights_only=True)

if os.path.exists(sharded_index):
    with open(sharded_index) as f:
        idx = json.load(f)
    shard_files = set()
    for k, v in idx["weight_map"].items():
        shard_files.add(v)
    all_state = {}
    for sf in sorted(shard_files):
        sf_path = os.path.join(input_dir, sf)
        if os.path.exists(sf_path):
            state = torch_load(sf_path)
            # Handle both bytes and str keys
            clean = {}
            for k, v in state.items():
                if isinstance(k, bytes):
                    k = k.decode('utf-8')
                if hasattr(v, 'numpy') or hasattr(v, 'shape'):
                    clean[k] = v
                elif isinstance(v, dict):
                    # Some checkpoints have nested dicts
                    for k2, v2 in v.items():
                        final_k = f"{k}.{k2}" if isinstance(k, str) else k
                        if hasattr(v2, 'shape'):
                            clean[final_k] = v2
            all_state.update(clean)
    st_save(all_state, out_dir + "/converted_model.safetensors")
    print(f"OK converted from {len(shard_files)} shards, {len(all_state)} tensors")
else:
    state = torch_load(single_bin)
    print(f"OK loaded {len(state)} tensors from {single_bin}")
    # Write as safetensors
    st_save(state, out_dir + "/converted_model.safetensors")
    print("OK converted to safetensors")
)PY";
        out.close();

        std::string cmd = "python3 " + script_path + " " + directory;
        int ret = std::system(cmd.c_str());
        std::error_code ec;
        fs::remove(script_path, ec);

        if (ret != 0) {
            throw std::runtime_error(
                "Failed to convert PyTorch .bin to safetensors in " + directory +
                ". Try: pip install torch safetensors " +
                "&& python -c 'from safetensors.torch import save_file; " +
                "import torch; state=torch.load(\"" + bin_path.string() + "\", map_location=\"cpu\"); " +
                "save_file(state, \"" + out_path + "\")\n");
        }

        // Retry loading the converted safetensors
        auto conv_path = fs::path(directory) / "converted_model.safetensors";
        if (fs::exists(conv_path)) {
            std::cerr << "[convert] Loaded converted safetensors\n";
            return load_safetensors(conv_path.string());
        }

        throw std::runtime_error(
            "Conversion completed but converted_model.safetensors not found in " + directory);
    }

    return all_weights;
}

void load_weights(
    const std::string& model_directory,
    std::unordered_map<std::string, mlx::core::array*>& weight_map)
{
    auto all_weights = load_safetensors_from_directory(model_directory);

    for (auto& [name, target_ptr] : weight_map) {
        auto it = all_weights.find(name);
        if (it != all_weights.end()) {
            *target_ptr = std::move(it->second);
        }
    }
}

} // namespace mlx_lm
