// SPDX-License-Identifier: Apache-2.0
// Experiment harness (lemon-mlx-engine P1): load one HSACO/CO in Redline AQL and
// replay once (correctness gate only — NOT gen t/s).
//
// API aligned with E2 harness + redline-dispatch examples (Runtime::initialize,
// Executable::load from bytes, SingleQueueBatchGraph).
//
// Build (out of tree against warpfront/redline checkout):
//   # copy or path-override this file as an example, OR:
//   cd /home/antmi/redline   # or /tmp/redline-warpfront
//   # add [[bin]] name=p1_load_hsaco path=... or cargo run --example after install
//   CARGO_TARGET_DIR=/tmp/redline-warpfront-target \
//   cargo run --release -p redline-dispatch --example p1_load_hsaco
//
// Env:
//   REDLINE_P1_HSACO   (required) path to .co / .hsaco
//   REDLINE_P1_SYMBOL  (default floor_k.kd)
//   REDLINE_P1_N       (default 2) dispatches per batch — API needs ≥2 for profiling
//   REDLINE_P1_WARMUP  (default 5)
//   REDLINE_P1_ITERS   (default 20)  // host µs only; ban as model TPS
//   REDLINE_P1_POLICY  (default BoundarySerialized)

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
    let s = std::env::var("REDLINE_P1_POLICY").unwrap_or_else(|_| "BoundarySerialized".into());
    match s.as_str() {
        "SystemEveryDispatch" | "system" => BatchFencePolicy::SystemEveryDispatch,
        "BoundaryIndependent" | "independent" => {
            eprintln!(
                "[p1] WARN: BoundaryIndependent forbidden for real decode (E1); \
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
    let hsaco = std::env::var("REDLINE_P1_HSACO").map_err(|_| {
        "REDLINE_P1_HSACO required (path to .co/.hsaco); see P1_PLAN.md"
    })?;
    let symbol = std::env::var("REDLINE_P1_SYMBOL").unwrap_or_else(|_| "floor_k.kd".into());
    // SingleQueueBatchGraph profiling path requires ≥2 dispatches (API InvalidBatchShape).
    let n = env_usize("REDLINE_P1_N", 2).max(2);
    let warmup = env_usize("REDLINE_P1_WARMUP", 5);
    let iters = env_usize("REDLINE_P1_ITERS", 20);
    let policy = policy_from_env();

    eprintln!("[p1] hsaco={hsaco}");
    eprintln!("[p1] symbol={symbol}");
    eprintln!("[p1] n={n} policy={policy:?} warmup={warmup} iters={iters}");

    let runtime = Runtime::initialize(load_symbols()?)?;
    let device = runtime.select_gpu(GpuSelector::Ordinal(0))?;
    eprintln!("[p1] device={}", device.name());

    let code: Arc<[u8]> = std::fs::read(&hsaco)?.into();
    let exec = Executable::load(&device, code)?;
    eprintln!("[p1] Executable::load OK ({} bytes)", std::fs::metadata(&hsaco)?.len());

    let pool = KernargPool::discover(&device)?;
    let geometry = LaunchGeometry::new([1, 1, 1], [1, 1, 1])?;
    let mut dispatches = Vec::with_capacity(n);
    for _ in 0..n {
        let kernel = exec.kernel(&symbol)?;
        let kernarg = pool.allocate_for(kernel.metadata())?;
        dispatches.push(RecordedDispatch::new(0, kernel, geometry, kernarg)?);
    }

    let range = device.queue_size_range();
    let want = ((n as u32).saturating_add(16)).next_power_of_two();
    let queue_size = want.clamp(*range.start(), *range.end());
    let mut graph = SingleQueueBatchGraph::create(
        &device,
        queue_size,
        dispatches,
        policy,
    )?;
    eprintln!("[p1] SingleQueueBatchGraph create OK (n={n})");

    for _ in 0..warmup {
        // SAFETY: graph owns kernels/kernargs/queue for its lifetime.
        unsafe {
            graph.replay_and_wait()?;
        }
    }
    eprintln!("[p1] warmup OK ({warmup})");

    let mut host_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _timing = unsafe { graph.replay_and_wait()? };
        host_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let med = median(host_us);
    // Host µs/replay only — DO NOT report as model gen t/s.
    eprintln!("[p1] host_median_us_per_replay={med:.3} (NOT gen t/s)");
    println!("P1_OK load+replay symbol={symbol} n={n} host_median_us={med:.3}");
    Ok(())
}
