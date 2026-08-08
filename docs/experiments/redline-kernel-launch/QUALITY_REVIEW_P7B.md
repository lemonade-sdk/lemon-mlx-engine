# Quality review — P7b full-gen L=1 sidecar verify

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Artifact:** [`P7B_FULLGEN_VERIFY.md`](P7B_FULLGEN_VERIFY.md) · `redline_decode_session.cpp` · `generate.cpp` · `logs/p7b-*-20260808-114354.err`

## Quintuple (simulated — spawn unavailable in fire)

### 1. explore — **PASS**
- Residual after P7: arm PASS inline only; full-gen L=1 acc never D2H-checked under model load.
- Local Qwen3.5-0.8B-4bit snapshot available; prior 401 was bad CLI (`--model` vs positional).

### 2. plan — **PASS**
- Option A (verify) over B (new product op) / D (gen A/B forbidden) / E.
- Add `maybe_redline_sidecar_verify`; wire `TokenIterator` dtor; smoke off / fullgen / xor.
- No `call_fn` replace; no default ON; label NOT gen t/s.

### 3. implement — **PASS**
- Verify: D2H `side_acc` vs `g_sidecar_expected` and `n(n+1)/2`.
- Build `chat` exit 0.
- Evidence: **fullgen PASS n=17 side_obs=153 side_exp=153**; off 0×; xor fail-closed.

### 4. quality-review — **PASS**
- Host correctness only; product Generation t/s line not claimed as redline A/B.
- Env exact `"1"` gates preserved.

### 5. supervisor — **PASS**

| Ban | Status |
|-----|--------|
| No LoopBrake / fake TPS | **OK** |
| No relabel host µs as gen t/s | **OK** (fullgen line is correctness, not t/s) |
| XOR fail-closed with pure-graph | **OK** |
| No HIP-graph re-enable | **OK** |
| Product default ON | **OK** (still OFF) |
| Evidence on branch logs | **OK** |

**Supervisor verdict: PASS**

## Residual
- Gen t/s A/B still **NOT RUN** (product path still owns `call_fn`).
- Toy `acc_k` only — not a real engine decode op.
