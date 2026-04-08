# CuPy FDTD — Complete Operation-by-Operation Breakdown

Every operation in the CuPy end-to-end FDTD simulation, in chronological order. This matches the code in `fdtd_cupy_deep_dive.ipynb` (Cell 11) exactly.

**Timer starts before Step 1. Timer ends after Step 6. Everything counted.**

---

## Step 1: Define CUDA Kernels [CPU only]

| # | Code | Runs on | Transfer | Blocks CPU? | What happens |
|---|------|---------|----------|-------------|--------------|
| 1 | `cp.RawKernel(source, 'update_H')` | **CPU** | None | No | **Compilation is LAZY.** Constructor only stores the source string and kernel name as Python objects. No CUDA compilation. No GPU memory. No GPU work at all. |
| 2 | `cp.RawKernel(source, 'update_Ez')` | **CPU** | None | No | Same — stores source string. |
| 3 | `cp.RawKernel(source, 'compute_energy')` | **CPU** | None | No | Same. Three Python objects created, zero GPU interaction. |

Actual CUDA compilation (nvrtc: C source → PTX → SASS machine code) happens at Step 5 on the **first kernel launch**. Takes ~500-1500ms. Compiled code is cached in `~/.cupy/kernel_cache/`.

---

## Step 2: Allocate GPU Memory [GPU alloc]

| # | Code | Runs on | Transfer | Blocks CPU? | What happens |
|---|------|---------|----------|-------------|--------------|
| 4 | `cp.zeros((200,200), dtype=cp.float32)` | **GPU** | None | No | `cudaMalloc` from CuPy's memory pool (160,000 bytes). `cudaMemsetAsync` fills with zeros — async, queued on GPU stream. Returns CuPy ndarray with `.data.ptr` pointing to GPU VRAM. |
| 5 | `cp.zeros(...)` for Hx_cu, Hy_cu | **GPU** | None | No | Two more 160 KB allocations + async memset. |
| 6 | `cp.zeros(1, dtype=cp.float32)` | **GPU** | None | No | 4 bytes for energy accumulator. |

**GPU memory after Step 2:** Ez (160 KB) + Hx (160 KB) + Hy (160 KB) + energy (4 B) = **480 KB on GPU VRAM**

---

## Step 3: Compute Pulse on CPU, Upload to GPU [CPU + H→D]

| # | Code | Runs on | Transfer | Blocks CPU? | What happens |
|---|------|---------|----------|-------------|--------------|
| 7 | `np.array([np.exp(...) for n in range(400)])` | **CPU** | None | Yes (CPU work) | Python loop calls `np.exp()` 400 times. Creates a 1,600-byte float32 array in CPU RAM. |
| 8 | `cp.array(pulse_np, dtype=cp.float32)` | **GPU** | **H→D** 1,600 bytes | Partial | `cudaMalloc` (1,600 bytes on GPU). Then: NumPy array → pinned host memory (page-locked) → `cudaMemcpyAsync(HostToDevice)` → GPU VRAM. Pinned alloc is sync on CPU; DMA transfer is async. |

---

## Step 4: Pre-cast Scalars & Configure Launch [CPU only]

| # | Code | Runs on | Transfer | Blocks CPU? | What happens |
|---|------|---------|----------|-------------|--------------|
| 9 | `np.float32(coeff_hx)` etc. (11 scalars) | **CPU** | None | No | Creates Python scalar objects. These stay on CPU. When passed to a kernel, they are copied **by value** into the kernel argument buffer (4 bytes each, not a GPU transfer). |
| 10 | `BLOCK = (8, 32)` | **CPU** | None | No | Python tuple for launch config. |
| 11 | `GRID = (25, 7)` | **CPU** | None | No | Ceiling division: `(200+7)//8=25`, `(200+31)//32=7`. |
| 12 | `total_energy_cu = np.zeros(n_steps)` | **CPU** | None | No | 400-element array on CPU for storing energy history. |

---

## Step 5: Main Simulation Loop (400 iterations) [GPU + SYNC + D→H every step]

This is the hot loop. **Every step** does the same work (energy computed every step to match NumPy):

| # | Code | Runs on | Transfer | Blocks CPU? | What happens |
|---|------|---------|----------|-------------|--------------|
| 13 | `cupy_update_H(GRID, BLOCK, (...))` | **GPU** | Args by value (~28 B) | **No** | CPU extracts `.data.ptr` from each CuPy array (reads a Python int attribute), packages raw device pointers + scalars into argument buffer, calls `cuLaunchKernel`. CPU returns in ~17μs. GPU queues the kernel. **First call triggers CUDA compilation (~500-1500ms).** Subsequent calls: ~3μs GPU-side. 44,800 threads each read 2 Ez floats, write 1 Hx and/or 1 Hy float. |
| 14 | `cupy_update_Ez(GRID, BLOCK, (..., np.int32(n)))` | **GPU** | Args by value (~44 B) | **No** | CPU creates `np.int32(n)` (4 bytes, CPU). Packages args, calls `cuLaunchKernel`. GPU waits for H kernel to finish (same-stream ordering), then runs. Each thread reads 4 H values + 1 pulse value, writes 1 Ez. Source: 1 thread reads `pulse_cu[n]` from GPU. PEC: ~800 edge threads write 0. |
| 15 | `energy_cu[:] = 0` | **GPU** | None | No | `cudaMemsetAsync` — zeros the 4-byte accumulator on GPU. Value is 0 so uses memset, not a kernel. Async. |
| 16 | `cupy_energy(GRID, BLOCK, (...))` | **GPU** | Args by value | **No** | 44,800 threads each compute cell energy (10 FLOPs: 3 loads, 4 multiplies, 2 adds, 1 atomicAdd). `atomicAdd` accumulates to the single float on GPU. |
| 17 | `cp.cuda.Device().synchronize()` | **CPU waits** | None | **Yes** | `cudaDeviceSynchronize()` — CPU blocks until ALL four operations above (H kernel, Ez kernel, memset, energy kernel) finish executing on GPU. **This stall happens every step.** |
| 18 | `float(energy_cu.get()[0])` | **D→H** | **4 bytes GPU→CPU** | **Yes** | `.get()` calls `cudaMemcpy(DeviceToHost)` — copies 4 bytes, syncs stream. Then `float()` extracts Python float. `.get()` itself syncs, but we already synced in step 17. |
| 19 | Store in `total_energy_cu[n]` | **CPU** | None | No | Writes the float into the CPU-side NumPy array. |

**Additionally, on 11 snapshot steps** (n = 0, 40, 80, ..., 360, 399):

| # | Code | Runs on | Transfer | Blocks CPU? | What happens |
|---|------|---------|----------|-------------|--------------|
| 20 | `cp.asnumpy(Ez_cu).copy()` | **D→H** | **160 KB GPU→CPU** | **Yes** | `cudaMemcpy(DeviceToHost)` — copies 200×200×4 = 160,000 bytes from GPU VRAM to a new NumPy array on CPU. Sync (waits for copy to complete). `.copy()` makes a separate CPU copy so it's not overwritten. |

---

## Step 6: Final Sync [SYNC]

| # | Code | Runs on | Transfer | Blocks CPU? | What happens |
|---|------|---------|----------|-------------|--------------|
| 21 | `cp.cuda.Device().synchronize()` | **CPU waits** | None | **Yes** | Final sync. Ensures all GPU work is done before stopping the wall-clock timer. |

**Timer stops here.**

---

## Transfer & Sync Accounting

### Per step (all 400 steps):
```
GPU kernels:  3 (H + Ez + energy)
Syncs:        1 (cudaDeviceSynchronize)
D→H:          4 bytes (energy value)
H→D:          0 bytes
CPU overhead:  ~50μs (3 launches × ~17μs each)
```

### Per snapshot step (11 out of 400):
```
(same as above, plus:)
D→H:          160,000 bytes (Ez field copy)
```

### Total for entire simulation:
```
Setup:
  H→D (one-time):      1,600 bytes (pulse array)
  GPU alloc:            ~482 KB

Simulation loop:
  GPU kernel launches:  400 × 3 = 1,200 total
  Syncs:                400 (every step, for energy readback)
  D→H (energy):        400 × 4 = 1,600 bytes
  D→H (snapshots):     11 × 160,000 = 1,760,000 bytes
  GPU memory I/O:       400 × ~1.8 MB = ~720 MB bandwidth

Total D→H:             ~1.72 MB
Total H→D:             ~1.6 KB (pulse only)
Total syncs:           401 (400 in loop + 1 final)
```

### Why this is slower than the "pure" version

The pure version (from the main notebook) only launches 2 kernels per step, syncs once at the very end. This end-to-end version syncs **400 times** because we read energy back to CPU every step. Each sync stalls the GPU pipeline for ~50-100μs.

```
Pure:          2 kernels/step × 400 + 1 sync      = 800 launches, 1 sync
End-to-end:    3 kernels/step × 400 + 400 syncs    = 1200 launches, 400 syncs
                                                      ^^^^^^^^^^^^^^^^^^^
                                                      ~20-40ms extra overhead
```

---

## Hidden Gotchas (verified from CuPy source code)

### `cupy_array[bool_mask] = 0.0` — Hidden synchronization!

Boolean fancy indexing internally calls `nonzero()` on the mask, which needs to count `True` elements — **copies count from GPU to CPU**. This is why we use branchless `Ez *= (1.0 - mask)` in GPU kernels instead of `Ez[mask] = 0`.

### `float(cupy_0d_array)` — Hidden synchronization!

`float()` on a CuPy scalar internally calls `.get()` → `cudaMemcpy(DeviceToHost)` + stream sync. We hit this 400 times (once per step for energy).

### `cp.max()` / `cp.sum()` — Do NOT sync by themselves

CuPy reductions return a 0-dimensional `cupy.ndarray` on the GPU, not a Python scalar. The reduction kernel runs async. Sync only happens when you extract the value via `float()`, `.item()`, or `.get()`.

### `arr[:] = 0` — Conditional behavior

Value 0 + C-contiguous array → `cudaMemsetAsync` (no kernel, just DMA, async).
Nonzero value or non-contiguous → launches a `fill_kernel` (still async).

### `cp.RawKernel()` constructor — Does NOT compile

Compilation is lazy — first `__call__()` triggers nvrtc. Cached on disk after first run.

---

## Quick Reference Table

| Operation | Transfer | Direction | GPU Alloc | Sync? | Kernel? |
|-----------|----------|-----------|-----------|-------|---------|
| `cp.zeros(shape)` | No | — | Yes (pool) | No | No (memsetAsync) |
| `cp.array(np_arr)` | Yes | CPU → GPU | Yes (pool) | Partial | No (DMA) |
| `RawKernel(src, name)` | No | — | No | No | No (lazy) |
| `rawkernel(grid, block, args)` | No | — | No | **No** | **Yes (async)** |
| `Device().synchronize()` | No | — | No | **Yes (all streams)** | No |
| `.get()` / `asnumpy()` | Yes | GPU → CPU | No (host) | **Yes** | No (DMA) |
| `arr[:] = 0` | No | — | No | No | Conditional |
| `arr[bool_mask] = 0` | Yes (hidden) | GPU → CPU | Yes (temp) | **Yes (hidden!)** | Yes (2+) |
| `float(cupy_scalar)` | Yes | GPU → CPU | No | **Yes (hidden!)** | No |
| `cp.max() / cp.sum()` | No | — | Yes (small) | No | Yes (reduction) |
| `cp.abs()` | No | — | Yes | No | Yes (elementwise) |

**Sources:** CuPy documentation (docs.cupy.dev), CuPy source code (github.com/cupy/cupy)
