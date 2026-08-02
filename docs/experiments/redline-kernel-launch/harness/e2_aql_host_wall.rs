// SPDX-License-Identifier: Apache-2.0
// Experiment harness (lemon-mlx-engine E2): host-wall timing of retained AQL
// multi-dispatch batch vs methodology of HIP N-launch wall.
// Env: REDLINE_FLOOR_HSACO (required), REDLINE_FLOOR_SYMBOL (floor_k.kd),
//      REDLINE_FLOOR_N (64), REDLINE_FLOOR_M (100), REDLINE_FLOOR_WARMUP (20)

use std::sync::Arc;
use std::time::Instant;

use redline_dispatch::aql::{
    BatchFencePolicy, Executable, GpuDevice, GpuSelector, KernargPool, LaunchGeometry,
    RecordedDispatch, Runtime, SingleQueueBatchGraph, load_symbols,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
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

fn build_graph(
    device: &GpuDevice,
    exec: &Executable,
    symbol: &str,
    n: usize,
    policy: BatchFencePolicy,
) -> Result<SingleQueueBatchGraph, Box<dyn std::error::Error>> {
    let pool = KernargPool::discover(device)?;
    let geometry = LaunchGeometry::new([1, 1, 1], [1, 1, 1])?;
    let mut dispatches = Vec::with_capacity(n);
    for _ in 0..n {
        let kernel = exec.kernel(symbol)?;
        let kernarg = pool.allocate_for(kernel.metadata())?;
        dispatches.push(RecordedDispatch::new(0, kernel, geometry, kernarg)?);
    }
    let range = device.queue_size_range();
    let want = ((n as u32).saturating_add(16)).next_power_of_two();
    let queue_size = want.clamp(*range.start(), *range.end());
    Ok(SingleQueueBatchGraph::create(
        device, queue_size, dispatches, policy,
    )?)
}

fn measure_host(
    device: &GpuDevice,
    exec: &Executable,
    symbol: &str,
    n: usize,
    m: usize,
    warmup: usize,
    policy: BatchFencePolicy,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    // returns (median host us/replay, median GPU span us if available)
    let mut graph = build_graph(device, exec, symbol, n, policy)?;
    for _ in 0..warmup {
        unsafe { graph.replay_and_wait()? };
    }
    let mut host = Vec::with_capacity(m);
    let mut gpu = Vec::with_capacity(m);
    for _ in 0..m {
        let t0 = Instant::now();
        let timing = unsafe { graph.replay_and_wait()? };
        host.push(t0.elapsed().as_secs_f64() * 1e6);
        gpu.push(timing.span_microseconds());
    }
    Ok((median(host), median(gpu)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hsaco = std::env::var("REDLINE_FLOOR_HSACO")
        .map_err(|_| "set REDLINE_FLOOR_HSACO")?;
    let symbol = std::env::var("REDLINE_FLOOR_SYMBOL").unwrap_or_else(|_| "floor_k.kd".to_owned());
    let n = env_usize("REDLINE_FLOOR_N", 64);
    let m = env_usize("REDLINE_FLOOR_M", 100);
    let warmup = env_usize("REDLINE_FLOOR_WARMUP", 20);

    let runtime = Runtime::initialize(load_symbols()?)?;
    let device = runtime.select_gpu(GpuSelector::Ordinal(0))?;
    println!(
        "e2_aql_host_wall: device={} N={n} M={m} warmup={warmup} hsaco={hsaco}",
        device.name()
    );
    let code: Arc<[u8]> = std::fs::read(&hsaco)?.into();
    let exec = Executable::load(&device, code)?;

    let policies = [
        ("SystemEveryDispatch", BatchFencePolicy::SystemEveryDispatch),
        ("BoundarySerialized", BatchFencePolicy::BoundarySerialized),
        ("BoundaryIndependent", BatchFencePolicy::BoundaryIndependent),
    ];

    println!(
        "  {:<22} {:>12} {:>12} {:>12} {:>12}",
        "policy", "host_us", "host_us/d", "gpu_us", "gpu_us/d"
    );
    let mut host_sys = 0.0;
    for (name, policy) in policies {
        let (host, gpu) = measure_host(&device, &exec, &symbol, n, m, warmup, policy)?;
        if name == "SystemEveryDispatch" {
            host_sys = host;
        }
        let vs = if host > 0.0 && host_sys > 0.0 {
            host_sys / host
        } else {
            0.0
        };
        println!(
            "  {name:<22} {host:>12.3} {:>12.4} {gpu:>12.3} {:>12.4}  vs_sys_host={vs:.3}x",
            host / n as f64,
            gpu / n as f64
        );
    }
    Ok(())
}
