// Tests for configuration parsing

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/base_config.h>
#include <mlx-lm/common/string_utils.h>
#include <nlohmann/json.hpp>

TEST_CASE("BaseConfiguration parsing", "[config]") {
    nlohmann::json j = {
        {"model_type", "llama"},
        {"eos_token_id", 2}
    };

    auto config = mlx_lm::parse_base_configuration(j);
    REQUIRE(config.model_type == "llama");
    REQUIRE(config.eos_token_ids.has_value());
    REQUIRE(config.eos_token_ids->values.size() == 1);
    REQUIRE(config.eos_token_ids->values[0] == 2);
}

TEST_CASE("BaseConfiguration with array EOS", "[config]") {
    nlohmann::json j = {
        {"model_type", "qwen2"},
        {"eos_token_id", {151645, 151643}}
    };

    auto config = mlx_lm::parse_base_configuration(j);
    REQUIRE(config.eos_token_ids->values.size() == 2);
}

TEST_CASE("BaseConfiguration text_config nested EOS (Qwen3.5 VLM)", "[config]") {
    nlohmann::json j = {
        {"model_type", "qwen3_5"},
        {"text_config", {
            {"eos_token_id", 248044},
            {"head_dim", 256}
        }}
    };

    auto config = mlx_lm::parse_base_configuration(j);
    REQUIRE(config.eos_token_ids.has_value());
    REQUIRE(config.eos_token_ids->values.size() == 1);
    REQUIRE(config.eos_token_ids->values[0] == 248044);
}

TEST_CASE("BaseConfiguration with quantization", "[config]") {
    nlohmann::json j = {
        {"model_type", "llama"},
        {"quantization", {
            {"group_size", 64},
            {"bits", 4}
        }}
    };

    auto config = mlx_lm::parse_base_configuration(j);
    REQUIRE(config.per_layer_quantization.has_value());
    REQUIRE(config.per_layer_quantization->default_quantization->group_size == 64);
    REQUIRE(config.per_layer_quantization->default_quantization->bits == 4);
}

TEST_CASE("BaseConfiguration with per-layer quantization", "[config]") {
    nlohmann::json j = {
        {"model_type", "llama"},
        {"quantization", {
            {"group_size", 64},
            {"bits", 4},
            {"model.embed_tokens", {
                {"group_size", 32},
                {"bits", 8}
            }},
            {"model.layers.0.self_attn.q_norm", false}
        }}
    };

    auto config = mlx_lm::parse_base_configuration(j);
    REQUIRE(config.per_layer_quantization.has_value());

    auto q = config.per_layer_quantization->quantization_for("model.embed_tokens");
    REQUIRE(q.has_value());
    REQUIRE(q->group_size == 32);
    REQUIRE(q->bits == 8);

    auto skip = config.per_layer_quantization->quantization_for("model.layers.0.self_attn.q_norm");
    REQUIRE(!skip.has_value());
}

// Port of Swift BaseConfigurationTests.testQuantization — verify that a simple
// quantization block yields the correct default and that per-layer lookup
// falls back to the default for an arbitrary layer name.
TEST_CASE("BaseConfiguration quantization default fallback", "[config]") {
    nlohmann::json j = nlohmann::json::parse(R"({
        "model_type": "Test",
        "quantization": {
            "group_size": 128,
            "bits": 4
        }
    })");

    auto config = mlx_lm::parse_base_configuration(j);
    REQUIRE(config.per_layer_quantization.has_value());

    // The default quantization should match the top-level values.
    auto& def = config.per_layer_quantization->default_quantization;
    REQUIRE(def.has_value());
    REQUIRE(def->group_size == 128);
    REQUIRE(def->bits == 4);

    // Any arbitrary layer should get the default quantization.
    auto q = config.per_layer_quantization->quantization_for("x");
    REQUIRE(q.has_value());
    REQUIRE(q->group_size == 128);
    REQUIRE(q->bits == 4);
}

// Port of Swift BaseConfigurationTests.testHeterogenousQuantization — verify
// per-layer overrides, false (skip), and true (use default) work correctly.
TEST_CASE("BaseConfiguration heterogeneous quantization", "[config]") {
    nlohmann::json j = nlohmann::json::parse(R"({
        "model_type": "Test",
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.embed_tokens": {
                "group_size": 32,
                "bits": 4
            },
            "model.layers.0.self_attn.q_norm": false,
            "true_layer": true
        }
    })");

    auto config = mlx_lm::parse_base_configuration(j);
    REQUIRE(config.per_layer_quantization.has_value());

    // The default quantization.
    auto& def = config.per_layer_quantization->default_quantization;
    REQUIRE(def.has_value());
    REQUIRE(def->group_size == 64);
    REQUIRE(def->bits == 4);

    SECTION("random layer gets default") {
        auto q = config.per_layer_quantization->quantization_for("x");
        REQUIRE(q.has_value());
        REQUIRE(q->group_size == 64);
        REQUIRE(q->bits == 4);
    }

    SECTION("layer with override gets custom quantization") {
        auto q = config.per_layer_quantization->quantization_for("model.embed_tokens");
        REQUIRE(q.has_value());
        REQUIRE(q->group_size == 32);
        REQUIRE(q->bits == 4);
    }

    SECTION("layer with false override is not quantized") {
        auto q = config.per_layer_quantization->quantization_for("model.layers.0.self_attn.q_norm");
        REQUIRE(!q.has_value());
    }

    SECTION("layer with true override gets default quantization") {
        auto q = config.per_layer_quantization->quantization_for("true_layer");
        REQUIRE(q.has_value());
        REQUIRE(q->group_size == 64);
        REQUIRE(q->bits == 4);
    }
}

TEST_CASE("StringOrNumber parsing", "[config]") {
    SECTION("string value") {
        nlohmann::json j = "llama3";
        mlx_lm::StringOrNumber sn;
        mlx_lm::from_json(j, sn);
        REQUIRE(sn.is_string());
        REQUIRE(sn.as_string() == "llama3");
    }

    SECTION("number value") {
        nlohmann::json j = 8.0;
        mlx_lm::StringOrNumber sn;
        mlx_lm::from_json(j, sn);
        REQUIRE(sn.is_float());
        REQUIRE(sn.as_float() == 8.0f);
    }
}
