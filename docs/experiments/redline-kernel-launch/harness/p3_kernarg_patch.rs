// SPDX-License-Identifier: Apache-2.0
// Experiment harness (lemon-mlx-engine P3): fixed micro-op = retained AQL batch
// + per-token in-place kernarg_u32 patch + replay.
//
// Models the graph_decode-style contract from P3_GRAPH_DECODE.md:
//   - Stable device buffer address packed into kernarg once at record time
//   - Per-"token" scalar patched via SingleQueueBatchGraph::patch_kernarg_u32
//   - No IB/graph rebuild between tokens
//
// Correctness: acc kernel atomicAdd(acc, val); n=2 dispatches share one acc;
// after T tokens with val=t each, expected = 2 * sum(1..=T).
//
// Metrics: host wall µs for (patch + replay_and_wait) only.
// DO NOT report as model gen t/s. Not a product wire.
//
// Build (out of tree against redline checkout):
//   cp docs/experiments/redline-kernel-launch/harness/p3_kernarg_patch.rs \
//     /home/antmi/redline/crates/redline-dispatch/examples/p3_kernarg_patch.rs
//   CARGO_TARGET_DIR=/tmp/redline-warpfront-target \
//     cargo build --release -p redline-dispatch --example p3_kernarg_patch
//
// Compile CO (gfx1150):
//   hipcc --genco --offload-arch=gfx1150 \
//     /home/antmi/redline/bench/acc_kernel.hip \
//     -o docs/experiments/redline-kernel-launch/logs/acc_kernel-gfx1150.co
//
// Env:
//   REDLINE_P3_HSACO   (required) path to acc_kernel .co
//   REDLINE_P3_SYMBOL  (default acc_k.kd)
//   REDLINE_P3_N       (default 2) dispatches; API needs ≥2 for profiling batch
//   REDLINE_P3_TOKENS  (default 64) patch+replay steps (T)
//   REDLINE_P3_WARMUP  (default 5)
//   REDLINE_P3_ITERS   (default 20) timed patch+replay after correctness T
//   REDLINE_P3_POLICY  (default BoundarySerialized)

use std::sync::Arc;
use std::time::Instant;

use redline_dispatch::aql::{
    BatchFencePolicy, Executable, GpuSelector, KernargPool, LaunchGeometry, RecordedDispatch,
    Runtime, SingleQueueBatchGraph, load_symbols,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn policy_from_env() -> BatchFencePolicy {
    let s = std::env::var("REDLINE_P3_POLICY").unwrap_or_else(|_| "BoundarySerialized".into());
    match s.as_str() {
        "SystemEveryDispatch" | "system" => BatchFencePolicy::SystemEveryDispatch,
        "BoundaryIndependent" | "independent" => {
            eprintln!(
                "[p3] WARN: BoundaryIndependent forbidden for real decode (E1); \
                 using BoundarySerialized"
            );
            BatchFencePolicy::BoundarySerialized
        }
        _ => BatchFencePolicy::BoundarySerialized,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hsaco = std::env::var("REDLINE_P3_HSACO").map_err(|_| {
        "REDLINE_P3_HSACO required (acc_kernel .co); see P3_MICRO_OP.md"
    })?;
    let symbol = std::env::var("REDLINE_P3_SYMBOL").unwrap_or_else(|_| "acc_k.kd".into());
    let n = env_usize("REDLINE_P3_N", 2).max(2);
    let tokens = env_usize("REDLINE_P3_TOKENS", 64).max(1);
    let warmup = env_usize("REDLINE_P3_WARMUP", 5);
    let iters = env_usize("REDLINE_P3_ITERS", 20);
    let policy = policy_from_env();

    eprintln!("[p3] hsaco={hsaco}");
    eprintln!("[p3] symbol={symbol}");
    eprintln!(
        "[p3] n={n} tokens={tokens} policy={policy:?} warmup={warmup} iters={iters}"
    );
    eprintln!("[p3] metric=host_us patch+replay (NOT gen t/s)");

    let runtime = Runtime::initialize(load_symbols()?)?;
    let device = runtime.select_gpu(GpuSelector::Ordinal(0))?;
    eprintln!("[p3] device={}", device.name());

    let code: Arc<[u8]> = std::fs::read(&hsaco)?.into();
    let exec = Executable::load(&device, code)?;
    eprintln!(
        "[p3] Executable::load OK ({} bytes)",
        std::fs::metadata(&hsaco)?.len()
    );

    let pool = KernargPool::discover(&device)?;
    let kernel0 = exec.kernel(&symbol)?;
    let kernarg_bytes = kernel0.metadata().kernarg_segment_size as usize;
    if kernarg_bytes < 12 {
        return Err(format!(
            "kernel {symbol} kernarg_segment_size={kernarg_bytes}; need ≥12 \
             ([acc:u64@0][val:u32@8])"
        )
        .into());
    }

    // Stable device accumulator — address baked once into every dispatch's kernarg
    // (graph_decode-style fixed buffer invariant).
    let mut acc = pool.allocate_executable_bytes(4)?;
    acc.as_mut_bytes().fill(0);
    let acc_addr = acc.address() as usize as u64;
    eprintln!("[p3] stable_acc_addr=0x{acc_addr:x} (baked once; not realloc per token)");

    let geometry = LaunchGeometry::new([1, 1, 1], [1, 1, 1])?;
    let mut dispatches = Vec::with_capacity(n);
    for _ in 0..n {
        let kernel = exec.kernel(&symbol)?;
        let mut kernarg = pool.allocate_for(kernel.metadata())?;
        let bytes = kernarg.as_mut_bytes();
        bytes.fill(0);
        bytes[0..8].copy_from_slice(&acc_addr.to_le_bytes());
        // val @8 starts 0; patched each token.
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes());
        dispatches.push(RecordedDispatch::new(0, kernel, geometry, kernarg)?);
    }

    let range = device.queue_size_range();
    let want = ((n as u32).saturating_add(16)).next_power_of_two();
    let queue_size = want.clamp(*range.start(), *range.end());
    let mut graph = SingleQueueBatchGraph::create(&device, queue_size, dispatches, policy)?;
    eprintln!("[p3] SingleQueueBatchGraph create OK (n={n})");

    // Correctness arm: T tokens, val = t on every dispatch, expected = n * sum(1..=T)
    for t in 1..=tokens {
        let val = t as u32;
        for d in 0..n {
            graph.patch_kernarg_u32(d, 8, val)?;
        }
        // SAFETY: graph owns kernels/kernargs/queue; acc buffer outlives graph.
        unsafe {
            graph.replay_and_wait()?;
        }
    }
    let observed = u32::from_le_bytes(acc.as_mut_bytes()[..4].try_into().unwrap());
    let sum_t = (tokens * (tokens + 1) / 2) as u64;
    let expected = (n as u64)
        .checked_mul(sum_t)
        .ok_or("expected overflow")? as u32;
    eprintln!("[p3] correctness observed={observed} expected={expected} (n*sum(1..T))");
    if observed != expected {
        eprintln!("P3_FAIL correctness mismatch observed={observed} expected={expected}");
        std::process::exit(2);
    }
    eprintln!("[p3] correctness PASS");

    // Reset acc for timed arm (patch val=0 would still leave residual; zero buffer).
    acc.as_mut_bytes().fill(0);

    for _ in 0..warmup {
        for d in 0..n {
            graph.patch_kernarg_u32(d, 8, 1)?;
        }
        unsafe {
            graph.replay_and_wait()?;
        }
    }
    eprintln!("[p3] warmup OK ({warmup})");

    // Timed: host wall of patch-all-dispatches + one replay (NOT gen t/s).
    let mut host_us = Vec::with_capacity(iters);
    for i in 0..iters {
        let val = ((i % 7) + 1) as u32;
        let t0 = Instant::now();
        for d in 0..n {
            graph.patch_kernarg_u32(d, 8, val)?;
        }
        let _timing = unsafe { graph.replay_and_wait()? };
        host_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let med = median(host_us);
    eprintln!("[p3] host_median_us_patch_plus_replay={med:.3} (NOT gen t/s)");
    println!(
        "P3_OK patch+replay symbol={symbol} n={n} tokens={tokens} \
         host_median_us={med:.3} correctness=PASS"
    );
    Ok(())
}
