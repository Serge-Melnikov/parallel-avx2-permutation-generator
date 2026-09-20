# Ultra-High-Throughput Parallel AVX2 Permutation Generator

A state-of-the-art SIMD-accelerated parallel algorithm for generating all n! permutations using a highly optimized variant of Shimon Even's method. The implementation utilizes 256-bit YMM registers to process split 128-bit lanes (Lane A and Lane B) simultaneously, achieving near-perfect linear scalability across physical CPU cores via OpenMP.

## Features
- **SIMD Architecture:** Vectorized lane splitting (2x 128-bit XMM inside YMM) for twin combinatorial search spaces.
- **Lock-Free Hot Path:** Register-level local XOR checksums completely eliminate false sharing and thread contention.
- **Exact Combinatorial Decoding:** `StructuralInitState` uses hardware-backed Bit Manipulation Instructions (`_pdep_u64` and `_tzcnt_u64`) to safely jump to any arbitrary start index up to n ≤ 16.
- **Practical Callback Interface:** Support for high-speed custom user callbacks (`PermCallback`) executed directly out of registers.
- **Dynamic Remainder Alignment:** Eliminates space allocation truncation errors under arbitrary thread counts.

## Performance & Scalability Benchmark

Evaluated on an x86_64 architecture with 6 physical cores (Hyper-Threading active):

| Space Complexity (n) | Thread Configuration | CPU Clock Cycles / Time | Speedup Factor | Status |
|-----------------------|----------------------|-------------------------|----------------|--------|
| n = 14              | `t1` (Single-Thread) | 17,419,771,736 cycles | Baseline (1.0x)| Verified |
| n = 14              | `t5` (Multi-Thread)  | **3,252,793,892** cycles | **5.3x** | Verified |
| n = 15              | `t1` (Single-Thread) | 98.0 seconds            | Baseline (1.0x)| Verified |
| n = 15              | `t6` (Physical Cores)| **16.0 seconds**        | **6.1x**       | Verified |
| n = 15              | `t12` (Hyper-Thread) | 16.0 seconds            | 6.1x (Saturated)| Port Starvation |

*Note: Hyper-Threading provides 0% additional gain because a single thread per physical core completely saturates the vector execution pipelines (SIMD shufflers).*

## Compilation

The program requires an x86_64 compiler supporting OpenMP and the AVX2 instruction set extension.

### GCC / Clang (Linux & MinGW)
Use aggressive optimization switches along with explicit architecture mapping:
```bash
gcc -O3 -march=native -fopenmp ymm_final_en_mt_cb.c -o ymm_final_en_mt_cb
```

### MSVC (Windows)
Ensure OpenMP and AVX2 switches are turned on in the compiler options:
```cmd
cl /O2 /arch:AVX2 /openmp ymm_final_en_mt_cb.c /Fe:ymm_final_en_mt_cb
```

## Usage

The executable expects exactly two arguments: the permutation target length (6 ≤ n ≤ 16) and the thread token configuration (e.g. `t5` or `5`).

```bash
# Execute benchmark for n=14 utilizing 5 parallel threads
./ymm_final_en_mt_cb 14 t5

# Generate and print all 10! permutations thread-safely across 7 workers
./ymm_final_en_mt_cb 10 t7 > output_permutations.txt
```

To enable the practical demonstration mode that invokes the `MyDemoCallback` routine and prints all generated states, uncomment the following line in the source file:
```c
#define HALF_PRINT_ENABLED
```
