# Ultra-High-Throughput Parallel AVX2 Permutation Generator

A state-of-the-art SIMD-accelerated parallel algorithm for generating all n! permutations using a highly optimized variant of Shimon Even's method. The implementation utilizes 256-bit YMM registers to process split 128-bit lanes (Lane A and Lane B) simultaneously, achieving near-perfect linear scalability across physical CPU cores via OpenMP.

## Features

* **Vector Ladder Propagation:** The engine operates on a compressed baseline space of n-1 elements. Once a baseline configuration is established, element n-1 acts as a fast-sweeping agent, propagating through the entire vector via ultra-fast SIMD shuffle loops (`_mm256_shuffle_epi8`). This yields n complete n-element permutations out of a single n-1 layout, reducing the amortized overhead to O(1) CPU cycles per permutation.
* **Combinatorial Reverse Invariance:** All vectorized routines are explicitly engineered to traverse **only the first half (n!/2)** of the total permutation space boundaries. The remaining half of the n! space is deterministically obtained by reversing each generated sequence from the first half. This **doubles the effective generation throughput** and **reduces the spatial memory footprint by exactly 50%**.
* **SIMD Twin-Lane Architecture:** Vectorized lane splitting (two 128-bit XMM tracks inside a single 256-bit YMM register) for processing independent combinatorial search spaces in parallel.
* **Lock-Free Hot Path:** Register-level local XOR checksums completely eliminate false sharing and thread contention across cores.
* **Exact Combinatorial Decoding:** `StructuralInitState` uses hardware-backed Bit Manipulation Instructions (`_pdep_u64` and `_tzcnt_u64`) along with bounded factorial lookup tables to safely jump to any arbitrary start index up to n ≤ 16.
* **Practical Callback Interface:** Support for high-speed custom user callbacks (`PermCallback`) executed thread-safely directly out of hardware registers.
* **Dynamic Remainder Alignment:** Completely eliminates space allocation truncation errors under arbitrary thread counts.

## Repository Structure

The project is structured into functional directories separating the baseline sequential algorithms from the high-performance parallel source files:

```text
├── sequential/
│   ├── p_opt_en.c              # Knuth's Algorithm P (Johnson-Trotter) optimized via isolated sweeping branch (3x speedup)
│   ├── ymm_final_en.c          # Single-threaded baseline vector implementation (idle run benchmark)
│   └── ymm_final_en1.c         # Practical single-threaded vector generator featuring a user callback
└── parallel/
    ├── ymm_final_en_mt.c       # Multi-threaded benchmark version (idle run, no callbacks)
    └── ymm_final_en_mt_cb.c    # Practical multi-threaded version featuring a user callback
```

#### Sequential Implementations (`sequential/`)
* **`p_opt_en.c` (Knuth's Algorithm P Optimized):** An accelerated implementation of Knuth's Algorithm P (the classic Johnson-Trotter adjacent transposition method) from Donald Knuth's *The Art of Computer Programming* (Volume 4A, Section 7.2.1.2). This routine achieves a **3x speedup** over the naive design by strategically decoupling the fast-sweeping ladder loops of element n-1 into an isolated, hyper-optimized execution branch, minimizing loop-overhead.
* **`ymm_final_en.c` (Single-Threaded SIMD Benchmark):** The sequential baseline leveraging 256-bit AVX2 vectors to execute split twin-lane combinatorial sweeps across the \(n!/2\) space boundaries. Stripped of function pointers, it measures raw hardware execution limits.
* **`ymm_final_en1.c` (Practical Single-Threaded SIMD):** Incorporates a sequential user-defined callback execution interface. It streams the first half of permutations straight from the YMM registers, allowing applications to process both the baseline and its reversed mirror layout dynamically.

#### Parallel Implementations (`parallel/`)
* **`ymm_final_en_mt.c` (Multi-Threaded SIMD Benchmark):** Utilizes OpenMP multi-core loops combined with our dynamic `StructuralInitState` space mapping decoder. It slices the \(n!/4\) invariant space into strict macro-periods for synchronous idle hardware benchmarks across physical cores.
* **`ymm_final_en_mt_cb.c` (Practical Multi-Threaded SIMD):** The definitive multi-threaded engine providing a thread-safe `PermCallback` pipeline. Each active worker thread streams distinct permutation slices out of its local registers into synchronized user handlers, enabling concurrent graph traversal or combinatorial optimization solvers with a 50% reduced memory footprint.

## Performance & Scalability Benchmark

Evaluated on an x86_64 architecture with 6 physical cores (Hyper-Threading active):

## Performance & Scalability Benchmark

Evaluated on an x86_64 architecture with 6 physical cores (Hyper-Threading active):

| Space Complexity (n) | Thread Configuration | CPU Clock Cycles / Time | Speedup Factor | Status |
|-----------------------|----------------------|-------------------------|----------------|--------|
| n = 14              | `t1` (Single-Thread) | 14.74B cycles           | Baseline (1.0x)| Verified |
| n = 14              | `t5` (Multi-Thread)  | **3.25B** cycles        | **4.53x**      | Verified |
| n = 15              | `t1` (Single-Thread) | 84.0 seconds            | Baseline (1.0x)| Verified |
| n = 15              | `t6` (Physical Cores)| **16.0 seconds**        | **5.25x**      | Verified |
| n = 15              | `t12` (Hyper-Thread) | 16.0 seconds            | 5.25x (Saturated)| Port Starvation |
| n = 16              | `t6` (Physical Cores)| **270.0 seconds**       | **4.98x**      | Thermal Drop* |

*Note: The scalability drop for n=16 (4.98x vs 5.25x) is induced by hardware thermal throttling. Sustained 100% AVX2 vector execution across all physical cores for 4.5 minutes triggers mobile CPU clock-frequency degradation to protect the silicon.*
*Hyper-Threading provides 0% additional gain because a single thread per physical core completely saturates the vector execution pipelines (SIMD shufflers).*

## Compilation

The program requires an x86_64 compiler supporting OpenMP and the AVX2 instruction set extension.

### GCC / Clang (Linux & MinGW)
Use aggressive optimization switches along with explicit architecture mapping:
```bash
gcc -O3 -march=native -fopenmp -funroll-loops ymm_final_en_mt_cb.c -o ymm_final_en_mt_cb
```

### MSVC (Windows)
Ensure OpenMP and AVX2 switches are turned on in the compiler options:
```cmd
cl /O2 /arch:AVX2 /openmp -funroll-loops ymm_final_en_mt_cb.c /Fe:ymm_final_en_mt_cb
```

## Usage

Both executable binaries expect exactly two arguments: the permutation target length (6 ≤ n ≤ 16) and the thread configuration token (e.g. `t5` or `5`).

### 1. High-Performance Benchmarking Mode (Default)
By default, both versions run in an accelerated idle configuration to measure raw hardware execution cycles without IO bottlenecks.
```bash
# Execute speed benchmark for n=14 utilizing 5 threads
./ymm_final_en_mt 14 t5
```

### 2. Verbose Output Mode (Printing All Permutations)
To enable the verbose generation trace that physically outputs the permutations to the console or a file, **you must uncomment the following line at the top of the respective source file (`ymm_final_en_mt.c` or `ymm_final_en_mt_cb.c`) before compilation**:

```c
#define HALF_PRINT_ENABLED
```

Once activated and compiled, use standard shell operators to safely redirect the massive stream of permutations into a structured text file:
```bash
# Generate and stream all 10! permutations thread-safely across 7 workers into a file
./ymm_final_en_mt_cb 10 t7 > output_permutations.txt
```
*Note: In verbose mode, the benchmark version (`ymm_final_en_mt`) prints the optimized baseline transitions, while the callback version (`ymm_final_en_mt_cb`) invokes `MyDemoCallback` to print the full expanded n-element permutations alongside their reverse mirror images.*
