// Architecture registration — load custom architectures from JSON files.

#include <mlx-lm/common/registry.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace mlx_lm {

void ArchitectureRegistry::load_from_file(const std::string& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("Architecture registration file not found: " + path);
    }

    std::ifstream f(path);
    nlohmann::json j;
    f >> j;

    // Accept either a single object or an array
    auto process_entry = [&](const nlohmann::json& entry) {
        ArchitectureRegistration arch;
        arch.model_type = entry.at("model_type").get<std::string>();
        arch.base_model = entry.value("base_model", std::string("llama"));

        if (entry.contains("key_remaps") && entry["key_remaps"].is_array()) {
            for (const auto& r : entry["key_remaps"]) {
                if (r.is_array() && r.size() == 2) {
                    arch.key_remaps.emplace_back(r[0].get<std::string>(), r[1].get<std::string>());
                }
            }
        }

        if (entry.contains("config_defaults") && entry["config_defaults"].is_object()) {
            for (auto& [key, val] : entry["config_defaults"].items()) {
                if (val.is_string()) {
                    arch.config_defaults[key] = val.get<std::string>();
                }
            }
        }

        if (entry.contains("skip_keys") && entry["skip_keys"].is_array()) {
            for (const auto& s : entry["skip_keys"]) {
                arch.skip_keys.push_back(s.get<std::string>());
            }
        }

        arch.activation_bits = entry.value("activation_bits", 0);
        arch.has_sub_norm = entry.value("has_sub_norm", false);

        std::cerr << "[arch] Registered: " << arch.model_type
                  << " -> " << arch.base_model
                  << " (" << arch.key_remaps.size() << " remaps"
                  << ", activation_bits=" << arch.activation_bits
                  << ", sub_norm=" << arch.has_sub_norm
                  << ")\n";

        register_architecture(arch);
    };

    if (j.is_array()) {
        for (const auto& entry : j) {
            process_entry(entry);
        }
    } else if (j.is_object()) {
        process_entry(j);
    } else {
        throw std::runtime_error("Invalid architecture registration file format");
    }
}

} // namespace mlx_lm
