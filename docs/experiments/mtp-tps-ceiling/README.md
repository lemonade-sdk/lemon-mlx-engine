# exp/mtp-tps-ceiling

Child of **`fix/mtp-stream-p0`** (previous product/experiment stack behind the tip).

**Purpose:** Execute `docs/analysis/mtp-review/06-tps-ceiling.md` §4 — re-probe `MLX_MTP_BATCH_VERIFY=1` on the post-fuse / P0-B stack + valid n_draft=3 row.

**Result:** **KILL** batch path; **plateau ~27 t/s** sequential n_draft=2. See **[RESULTS.md](./RESULTS.md)**.

**MTP_TIMING:** was `MTP_TIMING=1` on every run (hundreds of `[mtp-t]` lines in the four `S4_*.txt` logs).
