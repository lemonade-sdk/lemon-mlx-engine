// Convert + quantize a Qwen3.6-A3B (qwen3_5_moe) bf16/fp16 HF checkpoint to a
// combined MLX model that includes the MTP head, with per-component mixed
// precision. Runs on the engine's own (ROCm) MLX build — no Python required.
//
// The official Qwen/Qwen3.6-35B-A3B checkpoint already carries the MTP head
// inline (mtp.* tensors), so converting it directly yields a single one-file
// model with trunk + MTP head — loadable by lemon-mlx-engine's one-file path.
//
// Mixed precision rationale (see --help):
//   - experts / most linears: --bits (4/6/8) — the bulk of the size.
//   - router gate: --router-bits (default 8) — tiny but picks which experts run;
//     4-bit noise can flip borderline expert selection.
//   - lm_head: --lmhead-bits (default 8) — large vocab output projection.
//   - MTP head (mtp.*): --mtp-bits (default = --bits) — match the trunk so the
//     drafts and the (quantized) trunk agree; over-quantizing relative to the
//     trunk can LOWER acceptance.
//
// Usage:
//   convert <in_bf16_dir> <out_dir> --bits 4 [--router-bits 8] [--lmhead-bits 8]
//           [--mtp-bits N] [--group-size 64]

#include <mlx-lm/common/safetensors.h>
#include <mlx-lm/llm/models/qwen35_moe.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace mx = mlx::core;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Opts {
    std::string in_dir, out_dir;
    int bits = 4, router_bits = 8, lmhead_bits = 8, mtp_bits = -1, group_size = 64;
};

// Decide the bit width for a given tensor key, or -1 to leave it unquantized.
int bits_for(const std::string& key, const mx::array& w, const Opts& o) {
    // Only quantize 2D/3D matrices whose contraction dim is group-aligned.
    if (w.ndim() < 2) return -1;
    if (w.shape(-1) % o.group_size != 0 || w.shape(-1) < o.group_size) return -1;
    // Norms / conv / biases are excluded by the shape test above (1D) or by the
    // group-alignment test (conv kernel dim is tiny).
    bool is_mtp = key.rfind("mtp.", 0) == 0 || key.find(".mtp.") != std::string::npos;
    // The MTP head is a single-layer draft predictor with no depth to absorb
    // quant noise, and it's tiny (~0.5GB). mtp_bits<=0 keeps the whole head in
    // its source precision (bf16) — robust default for draft acceptance.
    if (is_mtp) return o.mtp_bits > 0 ? o.mtp_bits : -1;
    // Router gate is "...mlp.gate.weight" (NOT gate_proj / gate_up_proj).
    if (key.find(".gate.weight") != std::string::npos &&
        key.find("gate_proj") == std::string::npos &&
        key.find("gate_up") == std::string::npos)
        return o.router_bits;
    if (key == "lm_head.weight" || key.find("lm_head.weight") != std::string::npos)
        return o.lmhead_bits;
    return o.bits;
}

} // namespace

int main(int argc, char** argv) {
    Opts o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::stoi(argv[++i]); };
        if (a == "--bits") o.bits = next();
        else if (a == "--router-bits") o.router_bits = next();
        else if (a == "--lmhead-bits") o.lmhead_bits = next();
        else if (a == "--mtp-bits") o.mtp_bits = next();
        else if (a == "--group-size") o.group_size = next();
        else if (a == "--help" || a == "-h") {
            std::cerr << "Usage: convert <in_dir> <out_dir> --bits N "
                         "[--router-bits 8] [--lmhead-bits 8] [--mtp-bits N] "
                         "[--group-size 64]\n";
            return 0;
        }
        else pos.push_back(a);
    }
    if (pos.size() != 2) {
        std::cerr << "Need <in_dir> <out_dir>. See --help.\n";
        return 1;
    }
    o.in_dir = pos[0];
    o.out_dir = pos[1];
    if (o.mtp_bits < 0) o.mtp_bits = o.bits;  // match trunk by default

    std::cerr << "[convert] in=" << o.in_dir << " out=" << o.out_dir
              << " bits=" << o.bits << " router=" << o.router_bits
              << " lm_head=" << o.lmhead_bits << " mtp=" << o.mtp_bits
              << " group_size=" << o.group_size << "\n";

    // 1) Parse the source config and build the model (so sanitize() has the
    //    right architecture for the gate_up split + mtp stash).
    std::ifstream cfg_in(fs::path(o.in_dir) / "config.json");
    json cfg_json;
    cfg_in >> cfg_json;
    auto config = cfg_json.get<mlx_lm::Qwen35MoEConfiguration>();
    mlx_lm::Qwen35MoEModel model(config);

    // 2) Load bf16 weights.
    std::cerr << "[convert] loading bf16 weights...\n";
    auto weights = mlx_lm::load_safetensors_from_directory(o.in_dir);
    std::cerr << "[convert] loaded " << weights.size() << " tensors\n";

    // 2a) Raw-HF transforms that mlx_lm applies during conversion but the
    //     engine's sanitize does NOT (it expects mlx_lm's already-converted
    //     output): Qwen3-Next zero-centered RMSNorm (effective weight =
    //     stored + 1.0) and the conv1d weight axis layout. mlx-community
    //     checkpoints have these baked in; converting the raw official HF
    //     checkpoint must apply them or every norm/conv is wrong -> garbage.
    {
        static const std::vector<std::string> norm_sfx = {
            ".input_layernorm.weight", ".post_attention_layernorm.weight",
            "model.norm.weight", ".q_norm.weight", ".k_norm.weight",
            // MTP head's own zero-centered norms (verified comm = official+1.0):
            ".pre_fc_norm_embedding.weight", ".pre_fc_norm_hidden.weight",
            "mtp.norm.weight"};
        int nnorm = 0, nconv = 0;
        std::unordered_map<std::string, mx::array> fixed;
        fixed.reserve(weights.size());
        for (auto& [k, v] : weights) {
            mx::array w = v;
            if (k.find("conv1d.weight") != std::string::npos && w.ndim() >= 3 &&
                w.shape(-1) != 1) {
                w = mx::moveaxis(w, 2, 1);
                ++nconv;
            }
            bool is_norm = false;
            for (const auto& s : norm_sfx)
                if (k.size() >= s.size() &&
                    k.compare(k.size() - s.size(), s.size(), s) == 0) {
                    is_norm = true;
                    break;
                }
            if (is_norm && w.ndim() == 1) {
                w = mx::astype(
                    mx::add(mx::astype(w, mx::float32), mx::array(1.0f)),
                    w.dtype());
                ++nnorm;
            }
            fixed.emplace(k, std::move(w));
        }
        weights = std::move(fixed);
        std::cerr << "[convert] raw-HF fixups: +1.0 on " << nnorm
                  << " norms, moveaxis on " << nconv << " conv1d\n";
    }

    // 2b) sanitize (strips prefix, splits gate_up in bf16, stashes mtp.*).
    std::cerr << "[convert] sanitizing...\n";
    weights = model.sanitize(std::move(weights));

    // Re-attach the stashed mtp.* tensors (sanitize moves them out of `weights`)
    // so they get quantized + saved into the same combined checkpoint. The
    // stashed keys already carry the "mtp." prefix.
    for (const auto& [k, v] : model.mtp_weights()) {
        weights.emplace(k, v);
    }
    std::cerr << "[convert] trunk+mtp tensors to write: " << weights.size() << "\n";

    // 2c) Split the MTP head's COMBINED experts into the SPLIT switch_mlp form
    //     the head's SwitchGLU loader expects. The official checkpoint stores
    //     them as "mtp.layers.0.mlp.experts.gate_up_proj" [E, 2*moe_inter, hidden]
    //     + ".experts.down_proj" (note: no ".weight" suffix), but MTPHead's
    //     weight_map keys are "layers.0.mlp.switch_mlp.{gate,up,down}_proj.weight".
    //     sanitize() only splits the TRUNK experts (model.layers.*), never mtp.*,
    //     so without this the 256 head experts never load and draft acceptance is
    //     stuck at 0. Mirrors the trunk gate_up split in qwen35_moe sanitize().
    {
        const std::string gu = "mtp.layers.0.mlp.experts.gate_up_proj";
        auto gu_it = weights.find(gu);
        if (gu_it != weights.end()) {
            mx::array gate_up = std::move(gu_it->second);
            weights.erase(gu_it);
            int mid = gate_up.shape(-2) / 2;
            auto ndim = gate_up.ndim();
            mx::Shape start(ndim, 0);
            mx::Shape stop_gate(gate_up.shape().begin(), gate_up.shape().end());
            mx::Shape start_up(ndim, 0);
            mx::Shape stop_up(gate_up.shape().begin(), gate_up.shape().end());
            stop_gate[ndim - 2] = mid;
            start_up[ndim - 2] = mid;
            weights.insert_or_assign(
                "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
                mx::contiguous(mx::slice(gate_up, start, stop_gate)));
            weights.insert_or_assign(
                "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
                mx::contiguous(mx::slice(gate_up, start_up, stop_up)));
            auto dp_it = weights.find("mtp.layers.0.mlp.experts.down_proj");
            if (dp_it != weights.end()) {
                weights.insert_or_assign(
                    "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
                    std::move(dp_it->second));
                weights.erase(dp_it);
            }
            std::cerr << "[convert] split MTP head experts -> "
                         "switch_mlp.{gate,up,down}_proj.weight\n";
        }
    }

    // 3) Quantize per the mixed-precision policy; copy the rest as-is.
    std::unordered_map<std::string, mx::array> out;
    int nq = 0, nkeep = 0;
    for (auto& [key, w] : weights) {
        int b = bits_for(key, w, o);
        if (b <= 0) {
            out.emplace(key, w);
            ++nkeep;
            continue;
        }
        std::string base = key;
        const std::string suf = ".weight";
        if (base.size() > suf.size() &&
            base.compare(base.size() - suf.size(), suf.size(), suf) == 0)
            base = base.substr(0, base.size() - suf.size());
        // Slices from the gate_up split are strided views; make contiguous so
        // quantize reads the right elements.
        auto qr = mx::quantize(mx::contiguous(w), o.group_size, b);  // {wq, scales, biases}
        out.emplace(base + ".weight", qr[0]);
        out.emplace(base + ".scales", qr[1]);
        out.emplace(base + ".biases", qr[2]);
        ++nq;
    }
    std::cerr << "[convert] quantized " << nq << " weights, kept " << nkeep
              << " as-is; evaluating...\n";
    {
        std::vector<mx::array> all;
        for (auto& [k, v] : out) all.push_back(v);
        mx::eval(all);
    }

    // 4) Write a single shard + config.json (with quantization) + copy aux files.
    fs::create_directories(o.out_dir);
    std::cerr << "[convert] saving safetensors...\n";
    mx::save_safetensors((fs::path(o.out_dir) / "model.safetensors").string(), out,
                         {{"format", "mlx"}});

    // config.json: keep source config, add a quantization block (uniform-ish;
    // group_size + the dominant bits — per-tensor scales/biases carry the rest).
    cfg_json["quantization"] = {{"group_size", o.group_size}, {"bits", o.bits}};
    std::ofstream cfg_out(fs::path(o.out_dir) / "config.json");
    cfg_out << cfg_json.dump(2);
    cfg_out.close();

    for (const char* f : {"tokenizer.json", "tokenizer_config.json", "vocab.json",
                          "merges.txt", "special_tokens_map.json",
                          "generation_config.json", "chat_template.jinja"}) {
        auto src = fs::path(o.in_dir) / f;
        if (fs::exists(src)) fs::copy_file(src, fs::path(o.out_dir) / f,
                                           fs::copy_options::overwrite_existing);
    }
    std::cerr << "[convert] done -> " << o.out_dir << "\n";
    return 0;
}
