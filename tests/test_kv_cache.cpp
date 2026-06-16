// Tests for KV cache

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/kv_cache.h>
#include <mlx/mlx.h>

TEST_CASE("KVCacheSimple basic update", "[kvcache]") {
    mlx_lm::KVCacheSimple cache;
    REQUIRE(cache.offset() == 0);
    REQUIRE(!cache.max_size().has_value());

    auto keys = mlx::core::ones({1, 4, 3, 64});   // [B, heads, seq, dim]
    auto values = mlx::core::ones({1, 4, 3, 64});

    auto [k, v] = cache.update(keys, values);
    REQUIRE(cache.offset() == 3);
    REQUIRE(k.shape(2) == 3);
}

TEST_CASE("KVCacheSimple accumulates", "[kvcache]") {
    mlx_lm::KVCacheSimple cache;

    auto k1 = mlx::core::ones({1, 4, 5, 64});
    auto v1 = mlx::core::ones({1, 4, 5, 64});
    cache.update(k1, v1);
    REQUIRE(cache.offset() == 5);

    auto k2 = mlx::core::ones({1, 4, 3, 64});
    auto v2 = mlx::core::ones({1, 4, 3, 64});
    auto [k, v] = cache.update(k2, v2);
    REQUIRE(cache.offset() == 8);
    REQUIRE(k.shape(2) == 8);
}

TEST_CASE("KVCacheSimple trim", "[kvcache]") {
    mlx_lm::KVCacheSimple cache;

    auto keys = mlx::core::ones({1, 4, 10, 64});
    auto values = mlx::core::ones({1, 4, 10, 64});
    cache.update(keys, values);

    int trimmed = cache.trim(3);
    REQUIRE(trimmed == 3);
    REQUIRE(cache.offset() == 7);
}

TEST_CASE("RotatingKVCache basic", "[kvcache]") {
    mlx_lm::RotatingKVCache cache(16, 4);
    REQUIRE(cache.offset() == 0);
    REQUIRE(cache.max_size().value() == 16);
}

TEST_CASE("Type-erased KVCache", "[kvcache]") {
    mlx_lm::KVCache cache; // defaults to KVCacheSimple
    REQUIRE(cache.offset() == 0);
    REQUIRE(!cache.max_size().has_value());
    REQUIRE(cache.is_trimmable());
}

TEST_CASE("KVCacheSimple static reserve == legacy grow (in-place correctness)", "[kvcache]") {
    namespace mx = mlx::core;
    // Static cache: tiny initial capacity but a large reserve, so the FIRST
    // update sizes the buffer to cover the whole run and the grow-and-copy path
    // never executes. The legacy cache (default) grows by doubling. Both must
    // return byte-identical K/V — i.e. the static in-place writes are correct.
    mlx_lm::KVCacheSimple stat(8, 256);  // initial_capacity=8, reserve=256
    mlx_lm::KVCacheSimple dyn;           // legacy grow-by-doubling

    // "Prompt": 5 tokens of value 1.
    auto k1 = mx::full({1, 2, 5, 4}, 1.0f);
    stat.update(k1, k1);
    dyn.update(k1, k1);
    REQUIRE(stat.offset() == 5);

    // 20 single-token decode steps of value 2 (grow mode doubles 8->16->32 here).
    for (int i = 0; i < 20; ++i) {
        auto kt = mx::full({1, 2, 1, 4}, 2.0f);
        stat.update(kt, kt);
        dyn.update(kt, kt);
    }
    auto kt = mx::full({1, 2, 1, 4}, 3.0f);
    auto [skN, svN] = stat.update(kt, kt);
    auto [dkN, dvN] = dyn.update(kt, kt);

    REQUIRE(stat.offset() == 26);
    REQUIRE(skN.shape(2) == 26);
    REQUIRE(mx::all(mx::equal(skN, dkN)).item<bool>());  // static == legacy
    REQUIRE(mx::all(mx::equal(svN, dvN)).item<bool>());
    // The prompt prefix (positions 0..4) is preserved as value 1.
    auto prefix = mx::slice(skN, {0, 0, 0, 0}, {1, 2, 5, 4});
    REQUIRE(mx::all(mx::equal(prefix, mx::full({1, 2, 5, 4}, 1.0f))).item<bool>());
}

TEST_CASE("create_causal_mask", "[kvcache]") {
    auto mask = mlx_lm::create_causal_mask(4, 0);
    REQUIRE(mask.shape(0) == 4);
    REQUIRE(mask.shape(1) == 4);
}
