# Plan: Build BitNet Decode-Layer XCLBIN for NPU

## Goal
Build a full decode-layer xclbin for BitNet b1.58-2B-4T (ternary 2-bit packed) on the AMD Strix Halo NPU, following the same architecture as the working Q4NX decode layer.

## Model Architecture
- **BitNet b1.58-2B-4T**: 30 layers, hidden=2560, intermediate=6912, 20 heads, 5 KV heads
- **7 projections per layer**: Q[2560,2560], K[640,2560], V[640,2560], O[2560,2560], Gate[6912,2560], Up[6912,2560], Down[2560,6912]
- **Weight format**: uint8 packed ternary [out/4, in], single scalar `weight_scale` per projection
- **No per-projection RMS norms** (only block-level/sub-layer norms)

## Architecture (Copying Q4NX Decode Layer Pattern)

### Tile Grid
- **Main tiles**: 4×4 = 16 tiles (columns 2-5, rows 2-5) — same as Q4NX
- **Edge tiles**: Column 0 (attention), Column 1 (vectors), Columns 6-7 (SwiGLU, Down)
- **Total**: Same 40-tile layout as Q4NX decode layer, adapted for bitnet dimensions

### Phase Schedule
Same 7-phase pipeline as Q4NX:
1. **Q** projection: 20 chunks × 128 dim, 5 main16 blocks → 5 records per tile
2. **K** projection: 20 chunks × 128 dim → 1 record (5 KV heads)
3. **V** projection: same as K
4. **O** projection: 20 chunks × 128 dim → 5 records
5. **Gate** projection: 54 chunks × 128 dim → 13 records
6. **Up** projection: same as Gate
7. **Down** projection: 54 chunks × 128 dim, [2560, 6912] → 5 records per tile

### Weight Format (BitNet Ternary)
Instead of Q4NX's `{scale, offset, 4-bit data}` per group, BitNet uses:
- **Data**: uint8 packed, each byte holds 4 ternary values: `{0→-1, 1→0, 2→+1, 3→-1}`
- **Scale**: single bf16 `weight_scale` scalar per projection (replicated per output row)
- **No per-group zero-point/offset**

Total weight per tile projection = `(out_rows_per_tile/4) * in_dim * sizeof(uint8)` + `out_rows_per_tile * sizeof(bf16)`

### Chess Kernel (`bitnet_main16_ternary.o`)
A new Chess kernel that:
1. Loads 2-bit packed weight (4 values per byte)
2. Decodes ternary values: `{0→-1, 1→0, 2→+1}`  
3. Multiplies by weight_scale
4. MACs with activation BF16 → accumulates in BF16

This is simpler than the Q4NX kernel which needs per-group scale/offset.

### MLIR Generation
Copy `kernel_main16_q4nx_generate.py` → `kernel_main16_bitnet_generate.py`, adapting:
- Constants for 2560/6912 dims (instead of 4096/12288)
- Weight chunk size for 2-bit packed format
- Schedule constants (records per phase, chunks per record)

## Tasks

### Task 1: Constants and Contract (`bitnet_constants.h`, `bitnet_contract.py`)
Define the BitNet-specific constants:
- MAIN_ROWS_PER_TILE = 64 (smaller than Q4NX's? depends on tile config)
- CHUNK_DIM = 256 (same BF16 activation slice)
- WEIGHT_PACKING = 4 (4 ternary values per byte)
- Phase dims: Q/K/V/O/Gate/Up/Down

### Task 2: Chess Ternary Kernel (`bitnet_ternary_kernels.cc`)
Write the AIE kernel for 2-bit packed ternary matmul:
- `load_ternary_chunk()` — load uint8 packed data, decode to BF16
- `accum_ternary_chunk()` — MAC with activation
- `run_projection_body()` — same schedule pattern as Q4NX
- `bitnet_main16_layer_scheduler()` — phase dispatcher

### Task 3: Build MLIR Generator (`kernel_main16_bitnet_generate.py`)
Adapt the Python MLIR generator for BitNet:
- Tile grid placement
- DMA flows for activations/weights/records
- Runtime sequence (shim tile DMA)

### Task 4: Build the XCLBIN
- Compile Chess kernel → `.o`
- Generate MLIR → run `aiecc.py` → `.xclbin`
- Build for token capacity 127 (or 63 for smaller model)

### Task 5: Integrate into NPU Backend
- Add `bitnet` format to xclbin cache
- Modify weight loader to keep 2-bit packed format (don't dequant to BF16)
- Add `npu::matmul_bitnet()` to the unified plane API

### Task 6: Install and Test
- Install xclbin to `/usr/local/lib/npu/xclbins/`
- Test via `test_npu` with BitNet matmul
- Run server with BitNet model on NPU

## Timeline
- Tasks 1-2: 4-5 hours (Chess kernel is the hardest part)
- Task 3: 1-2 hours
- Task 4: 1 hour (build + debug)
- Tasks 5-6: 1-2 hours
- **Total: 7-10 hours (1-1.5 days of focused work)**
