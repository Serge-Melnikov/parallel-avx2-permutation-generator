/**
 * @file ymm_final_en_mt.c
 * @brief Ultra-High-Throughput Parallel AVX2 Permutation Generator 
 *
 * @copyright Copyright (c) 2026 Serge Melnikov. All rights reserved.
 * @license This project is licensed under the MIT License - see the LICENSE file for details.
 *
 * @details
 * - Core Architecture: Implements a highly optimized variant of Shimon Even's method,
 *   leveraging 256-bit AVX2 registers split into independent dual 128-bit execution lanes.
 * - Logic: Utilizes an ultra-fast vector byte shuffling strategy (_mm256_shuffle_epi8)
 *   and loop-invariant factorials to achieve O(1) amortized cycles-per-permutation overhead.
 * - Engineering Features:
 *   - Combinatorial Reverse Invariance upping throughput by 2x and slashing memory footprint by 50%.
 *   - Lock-free thread-local register accumulation completely eliminating cache-line false sharing.
 *   - Hardware-accelerated absolute state decoder (StructuralInitState) via native _pdep_u64.
 *
 * @environment
 * - Platform: Pure Cross-platform compliant (Windows / Linux) topology execution.
 * - Compiler: GCC / MinGW / MSVC (Best performance achieved via -O3 -march=native -fopenmp -funroll-loops -s -Wall).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <immintrin.h>
#include <omp.h>

#define MAXN 16

/* Cross-compiler alignment configuration for GCC, Clang, and MSVC */
#if defined(_MSC_VER)
    #define ALIGN_32 __declspec(align(32))
    #define ALIGN_16 __declspec(align(16))
#elif defined(__GNUC__) || defined(__clang__)
    #define ALIGN_32 __attribute__((aligned(32)))
    #define ALIGN_16 __attribute__((aligned(16)))
#else
    #define ALIGN_32
    #define ALIGN_16
#endif

/* Cross-compiler loop unrolling macros */
#if defined(__clang__)
    #define UNROLL_16 _Pragma("clang loop unroll_count(16)")
#elif defined(__GNUC__) && (__GNUC__ >= 5)
    #define UNROLL_16 _Pragma("GCC unroll 16")
#elif defined(_MSC_VER)
    #define UNROLL_16 _Pragma("loop(no_vector)")
#else
    #define UNROLL_16
#endif

/* Cross-compiler trailing zero count configuration safe for MSVC, GCC, and Clang */
#if defined(_MSC_VER)
    #include <intrin.h>
    static inline uint64_t TZCNT_U64(uint64_t value) 
    {
        unsigned long index;
        if (_BitScanForward64(&index, value)) return (uint64_t)index;
        return 64;
    }
#elif defined(__GNUC__) || defined(__clang__)
    static inline uint64_t TZCNT_U64(uint64_t value) 
    {
        if (value == 0) return 64;
        return (uint64_t)__builtin_ctzll(value);
    }
#endif

/* Cross-compiler optimization barrier configuration */
#if defined(_MSC_VER)
    #include <intrin.h>
    #pragma intrinsic(_ReadWriteBarrier)
    #define COMPILER_BARRIER() _ReadWriteBarrier()
#elif defined(__GNUC__) || defined(__clang__)
    #define COMPILER_BARRIER() __asm__ volatile("" : : "g" (&ymmGlobalChecksum) : "memory")
#else
    #define COMPILER_BARRIER()
#endif

// Cross-platform header for CPUID
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#include <cpuid.h>
#endif

typedef int32_t I32;
typedef uint64_t U64;
typedef int64_t I64;
typedef uint32_t U32;
typedef uint16_t U16;
typedef uint8_t U8;

/* Uncomment to enable verbose output */
// #define HALF_PRINT_ENABLED
#define PRINT_TIME 1

/* Shared read-only array for vector masks used by all threads */
ALIGN_32 __m256i ymmMasks[16];

/* Global execution control variables */
I64 n = 0;
I64 hiddenZero = 0;

/* Global register to accumulate ladder vector checksums via lightweight atomic XOR */
ALIGN_32 volatile __m256i ymmGlobalChecksum = {0};

/* Variable to accumulate the runtime vector checksum trace if printing is disabled */
volatile I64 liveChecksumAccum = 0;

void PrintHalfLanes(const U8* lanes, I32 n_val)
{
  static U8 chars[] = "0123456789abcdef";
  const U8* pA = lanes;
  const U8* pB = lanes + 16;
  for (I32 i = 0; i < n_val; i++) printf("%c", chars[pA[i]]);
  puts("");
  for (I32 i = 0; i < n_val; i++) printf("%c", chars[pB[i]]);
  puts("");
}

/* Highly optimized lock-free ladder processing function writing to thread-local register accumulator */
static inline void ProcessExtendedLadder(I32 n_val, __m256i currentYMM, int thread_id, I64 thread_quanta, __m256i *localChecksum)
{
  /* 1. Inject element n (which is n_val) at the head (1-byte left shift and bitwise OR) */
  __m256i shifted = _mm256_bslli_epi128(currentYMM, 1);
  __m256i n_vector = _mm256_setr_epi8(
      (char)n_val, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      (char)n_val, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0
  );
  __m256i ladderYMM = _mm256_or_si256(shifted, n_vector);

  /* PRINT: Atomically output the full 6-element permutation from both lanes */
  #ifdef HALF_PRINT_ENABLED
  {
      ALIGN_32 U8 localBuf[32];
      static U8 chars[] = "0123456789abcdef";
      char buf[256];

      _mm256_store_si256((__m256i*)localBuf, ladderYMM);
      int offset = sprintf(buf, "[T%d | #%" PRId64 "]: ", thread_id, (I64)thread_quanta);
      for (I32 i = 0; i <= n_val; i++) offset += sprintf(buf + offset, "%c", chars[localBuf[i]]);
      offset += sprintf(buf + offset, " | ");
      for (I32 i = 0; i <= n_val; i++) offset += sprintf(buf + offset, "%c", chars[localBuf[i + 16]]);
      sprintf(buf + offset, "\n");

      _Pragma("omp critical")
      {
          printf("%s", buf);
          fflush(stdout);
      }
  }
  #else
  if (__builtin_expect(!!(hiddenZero != 0), 0))
  {
      ALIGN_32 U8 localBuf[32];
      _mm256_store_si256((__m256i*)localBuf, ladderYMM);
      liveChecksumAccum += localBuf[hiddenZero];
  }
  #endif

  /* 2. Propagate element n through the entire permutation (shuffle steps) */
  UNROLL_16
  for (I64 step = 0; step < n_val; step++)
  {
     ladderYMM = _mm256_shuffle_epi8(ladderYMM, ymmMasks[step]);

     #ifdef HALF_PRINT_ENABLED
     {
         ALIGN_32 U8 localBuf[32];
         static U8 chars[] = "0123456789abcdef";
         char buf[256];

         _mm256_store_si256((__m256i*)localBuf, ladderYMM);
         int offset = sprintf(buf, "[T%d | #%" PRId64 "]: ", thread_id, thread_quanta);
         for (I32 i = 0; i <= n_val; i++) offset += sprintf(buf + offset, "%c", chars[localBuf[i]]);
         offset += sprintf(buf + offset, " | ");
         for (I32 i = 0; i <= n_val; i++) offset += sprintf(buf + offset, "%c", chars[localBuf[i + 16]]);
         sprintf(buf + offset, "\n");

         _Pragma("omp critical")
         {
             printf("%s", buf);
             fflush(stdout);
         }
     }
     #endif
  }

  /* FAST REGISTER UPDATE: Zero locks inside the critical sweep path */
  *localChecksum = _mm256_xor_si256(*localChecksum, ladderYMM);
}

/* Mask array initialization - shared across all OpenMP threads */
void PreparePermYMM_Masks(I32 n_val)
{
  for (I32 i = 0; i < n_val; i++)
  {
     ALIGN_16 char tempMask[16];
     for (I32 j = 0; j < 16; j++) tempMask[j] = (char)j;
     tempMask[i] = (char)(i + 1);
     tempMask[i + 1] = (char)i;

     __m128i xmmM = _mm_load_si128((const __m128i*)tempMask);
     ymmMasks[i] = _mm256_setr_m128i(xmmM, xmmM);
  }
}

/* Truly mathematical state decoder mapping absolute indices to exact permutation spaces */
void StructuralInitState(I64 total_n, I64 start_num, I64 *p, I64 *ind, I64 *dir, I64 *act)
{
  I64 nminus1 = total_n - 1;

  if (start_num == 0)
  {
    for (I64 i = 0; i < total_n; i++) ind[i] = p[i] = i;
    *dir = 0;
    *act = -2;
    return;
  }

  I64 fact[MAXN + 2];
  fact[0] = 1;
  for (int i = 1; i <= total_n; i++) fact[i] = fact[i - 1] * i;

  U64 free_mask = (1ULL << total_n) - 1;
  U64 r = start_num;

  for (int m = (int)total_n; m >= 1; m--)
  {
    U64 q = r / m;
    int rem = (int)(r % m);
    int step_val = (q % 2 == 0) ? ((m - 1) - rem) : rem;

    U64 bit_pos = _pdep_u64(1ULL << step_val, free_mask);
    I64 physical_idx = TZCNT_U64(bit_pos);

    p[physical_idx] = (I64)(m - 1);
    free_mask &= ~bit_pos;
    r = q;
  }

  for (I64 i = 0; i < total_n; i++) ind[p[i]] = i;
  *dir = 0;
  *act = 0;

  for (I64 i = 0; i < nminus1; i++)
  {
    I64 L = fact[total_n] / fact[i];
    I64 rem = start_num % (2 * L);
    I64 delta = fact[total_n] / fact[i + 1];

    I64 is_right = 0;
    I64 is_active = 0;

    if (i > 0)
    {
      if (rem < L)
      {
        is_right = 0;
        if (rem < L - delta) is_active = 1;
      }
      else
      {
        is_right = 1;
        I64 rem2 = rem - L;
        if (rem2 < L - delta) is_active = 1;
      }
    }
    I64 bit_shift = total_n - 1 - i;
    if (is_right) *dir |= (1ULL << bit_shift);
    if (is_active) *act |= (1ULL << bit_shift);
  }
}

/* Parallel OpenMP worker executing at full core speed via localized trace logic */
void GenPermYMM_Half_Parallel(I32 n_val, int thread_id, I64 threadTargetPerm, I64 startNumA, I64 startNumB)
{
  I64 nminus1 = (I64)n_val - 1, step[2] = {-1, 1}, mask[MAXN], bitPos[MAXN];

  /* --- Thread-Local Hardware Accumulator --- */
  ALIGN_32 __m256i threadLocalChecksum = _mm256_setzero_si256();

  /* --- Lane A Local Control Variables --- */
  ALIGN_16 I64 indA[MAXN] = {0};
  ALIGN_16 I64 pA_raw[MAXN] = {0};
  I64 scDirA = 0, scActA = 0;

  /* --- Lane B Local Control Variables --- */
  ALIGN_16 I64 indB[MAXN] = {0};
  ALIGN_16 I64 pB_raw[MAXN] = {0};
  I64 scDirB = 0, scActB = 0;

  /* Decode absolute combinatorial vectors cleanly */
  StructuralInitState((I64)n_val, startNumA, pA_raw, indA, &scDirA, &scActA);
  StructuralInitState((I64)n_val, startNumB, pB_raw, indB, &scDirB, &scActB);

  I32 dirA = (I32)scDirA; I32 actA = (I32)scActA;
  I32 dirB = (I32)scDirB; I32 actB = (I32)scActB;

  ALIGN_16 char initA[16] = {0};
  ALIGN_16 char initB[16] = {0};
  for (I32 i = 0; i < n_val; i++)
  {
      initA[i] = (char)pA_raw[i];
      initB[i] = (char)pB_raw[i];
  }

  ALIGN_32 __m256i currentYMM = _mm256_setr_m128i(_mm_load_si128((const __m128i*)initA), _mm_load_si128((const __m128i*)initB));
  ALIGN_32 U8 pLanes[32];

  /* Pre-calculate structural shift parameters */
  for (I32 i = 0; i < n_val; i++)
  {
      bitPos[i] = 1ULL << (nminus1 - i);
      mask[i] = bitPos[i] - 1ULL;
  }

  I64 totalCount = 0;
  ProcessExtendedLadder(n_val, currentYMM, thread_id, totalCount, &threadLocalChecksum);
  totalCount++;

  while (1)
  {
     /* --- LEFTWARD SWEEP --- */
     for (I32 i = nminus1; i > 0; i--)
     {
       currentYMM = _mm256_shuffle_epi8(currentYMM, ymmMasks[i - 1]);
       ProcessExtendedLadder(n_val, currentYMM, thread_id, totalCount, &threadLocalChecksum);
     }
     totalCount += nminus1;

     _mm256_store_si256((__m256i*)pLanes, currentYMM);
     U8* pA = pLanes;
     U8* pB = pLanes + 16;

     I64 curItemA = nminus1 - TZCNT_U64(actA);
     I64 curItemB = nminus1 - TZCNT_U64(actB);

     /* --- INTERSECTION 1: LANE A --- */
     I64 curIndA = indA[curItemA] + 1;
     I64 curDirA = step[(dirA >> (nminus1 - curItemA)) & 1];
     I64 targetIndA = curIndA + curDirA;
     I32 maskIdxA = curIndA < targetIndA ? (I32)curIndA : (I32)targetIndA;

     /* --- INTERSECTION 1: LANE B --- */
     I64 curIndB = indB[curItemB] + 1;
     I64 curDirB = step[(dirB >> (nminus1 - curItemB)) & 1];
     I64 targetIndB = curIndB + curDirB;
     I32 maskIdxB = curIndB < targetIndB ? (I32)curIndB : (I32)targetIndB;

     __m128i lowTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 0), _mm256_castsi256_si128(ymmMasks[maskIdxA]));
     __m128i highTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 1), _mm256_castsi256_si128(ymmMasks[maskIdxB]));
     currentYMM = _mm256_setr_m128i(lowTransformed, highTransformed);

     ProcessExtendedLadder(n_val, currentYMM, thread_id, totalCount, &threadLocalChecksum);
     totalCount++;

     I64 itemDisplacedA = pA[targetIndA];
     indA[curItemA] = targetIndA - 1; indA[itemDisplacedA] = curIndA - 1;

     I64 itemDisplacedB = pB[targetIndB];
     indB[curItemB] = targetIndB - 1; indB[itemDisplacedB] = curIndB - 1;

     curIndA += curDirA;
     dirA ^= (I32)mask[curItemA];
     actA |= (I32)mask[curItemA]; actA ^= 1;
     I64 isBlockedA = 1;
     I64 nextA = curIndA + curDirA;
     if (nextA >= 1 && nextA <= nminus1 && curItemA > pA[nextA]) isBlockedA = 0;
     if (isBlockedA) actA &= ~(I32)bitPos[curItemA];

     curIndB += curDirB;
     dirB ^= (I32)mask[curItemB];
     actB |= (I32)mask[curItemB]; actB ^= 1;
     I64 isBlockedB = 1;
     I64 nextB = curIndB + curDirB;
     if (nextB >= 1 && nextB <= nminus1 && curItemB > pB[nextB]) isBlockedB = 0;
     if (isBlockedB) actB &= ~(I32)bitPos[curItemB];

     /* --- RIGHTWARD SWEEP --- */
     for (I32 i = 0; i < nminus1; i++)
     {
       currentYMM = _mm256_shuffle_epi8(currentYMM, ymmMasks[i]);
       ProcessExtendedLadder(n_val, currentYMM, thread_id, totalCount, &threadLocalChecksum);
     }
     totalCount += nminus1;
     if (totalCount >= threadTargetPerm) break;

     _mm256_store_si256((__m256i*)pLanes, currentYMM);

     curItemA = nminus1 - TZCNT_U64(actA);
     curItemB = nminus1 - TZCNT_U64(actB);

     /* --- INTERSECTION 2: LANE A --- */
     curIndA = indA[curItemA];
     curDirA = step[(dirA >> (nminus1 - curItemA)) & 1];
     targetIndA = curIndA + curDirA;
     maskIdxA = curIndA < targetIndA ? (I32)curIndA : (I32)targetIndA;

     /* --- INTERSECTION 2: LANE B --- */
     curIndB = indB[curItemB];
     curDirB = step[(dirB >> (nminus1 - curItemB)) & 1];
     targetIndB = curIndB + curDirB;
     maskIdxB = curIndB < targetIndB ? (I32)curIndB : (I32)targetIndB;

     lowTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 0), _mm256_castsi256_si128(ymmMasks[maskIdxA]));
     highTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 1), _mm256_castsi256_si128(ymmMasks[maskIdxB]));
     currentYMM = _mm256_setr_m128i(lowTransformed, highTransformed);

     ProcessExtendedLadder(n_val, currentYMM, thread_id, totalCount, &threadLocalChecksum);
     totalCount++;

     itemDisplacedA = pA[targetIndA];
     indA[curItemA] = targetIndA; indA[itemDisplacedA] = curIndA;

     itemDisplacedB = pB[targetIndB];
     indB[curItemB] = targetIndB; indB[itemDisplacedB] = curIndB;

     curIndA += curDirA;
     dirA ^= (I32)mask[curItemA];
     actA |= (I32)mask[curItemA]; actA ^= 1;
     isBlockedA = 1; nextA = curIndA + curDirA;
     if (nextA >= 0 && nextA <= nminus1 && curItemA > pA[nextA]) isBlockedA = 0;
     if (isBlockedA) actA &= ~(I32)bitPos[curItemA];

     curIndB += curDirB;
     dirB ^= (I32)mask[curItemB];
     actB |= (I32)mask[curItemB]; actB ^= 1;
     isBlockedB = 1;
     nextB = curIndB + curDirB;
     if (nextB >= 0 && nextB <= nminus1 && curItemB > pB[nextB]) isBlockedB = 0;
     if (isBlockedB) actB &= ~(I32)bitPos[curItemB];
  }

  /* Single localized thread merge once per core task lifetime to completely prevent false sharing */
  _Pragma("omp critical")
  {
      ymmGlobalChecksum = _mm256_xor_si256((__m256i)ymmGlobalChecksum, threadLocalChecksum);
  }
}

int main(int argc, char* argv[])
{
  U64 count1, count2;
  U32 cpuInfo[4];
  int num_threads = 1;

  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <n> <tThreads>\n", argv[0]);
    return 1;
  }

  n = atoi(argv[1]);
  if (n < 6 || n > 16)
  {
    fputs("Error: wrong parameter n (must be 6..16)\n", stderr);
    return 1;
  }
  --n;

  if (argv[2][0] == 't' || argv[2][0] == 'T')
    num_threads = atoi(&argv[2][1]);
  else
    num_threads = atoi(argv[2]);

  if (num_threads < 1)
  {
      fputs("Error: thread count must be at least 1\n", stderr);
      return 1;
  }

  hiddenZero = *(volatile I64*)&hiddenZero;

  volatile int warm_dummy = 42;
  for (int i = 0; i < 50000000; ++i) warm_dummy = warm_dummy * 3 + i;
  PreparePermYMM_Masks((I32)n);

  /* Compute global boundaries before threads start */
  I64 fact[MAXN + 2];
  fact[0] = 1;
  for (int i = 1; i <= n; i++) fact[i] = fact[i - 1] * i;
  I64 totalSubSpace = fact[n] / 4;
  I64 macroPeriod = 2 * n;
  I64 totalPeriods = totalSubSpace / macroPeriod;
  if (num_threads > totalPeriods) num_threads = (int)totalPeriods;

  /* Set target concurrency context for upcoming OpenMP blocks */
  omp_set_num_threads(num_threads);

  /* Serialize and record benchmark start time using CPUID and TSC counters */
#ifdef _MSC_VER
  __cpuid(cpuInfo, 0);
#else
  __get_cpuid(0, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
#endif
  _mm_lfence();
  count1 = __rdtsc();
  _mm_lfence();

  #pragma omp parallel
  {
      int tid = omp_get_thread_num();

      I64 quantaPerThread = totalSubSpace / num_threads;
      /* Align the chunk strictly to the macro period boundary */
      quantaPerThread = (quantaPerThread / macroPeriod) * macroPeriod;
      if (quantaPerThread == 0) quantaPerThread = macroPeriod;

      I64 startNumA = (I64)tid * quantaPerThread;
      I64 startNumB = totalSubSpace + startNumA;

      I64 threadTarget = quantaPerThread;
      /* The last active thread dynamically absorbs the remaining tail periods */
      if (tid == num_threads - 1) threadTarget = totalSubSpace - startNumA;

      GenPermYMM_Half_Parallel((I32)n, tid, threadTarget, startNumA, startNumB);
  }

  /* Serialize and record benchmark end time */
  _mm_lfence();
  count2 = __rdtsc();
  _mm_lfence();

  if (PRINT_TIME == 1)
  {
     #ifndef HALF_PRINT_ENABLED
     printf("%" PRIu64 " CPU cycles\n", count2 - count1);
     #endif
  }

  COMPILER_BARRIER();
  ALIGN_32 char finalBuf[32];
  _mm256_store_si256((__m256i*)finalBuf, (__m256i)ymmGlobalChecksum);
  I64 totalSum = 0;
  for (I32 i = 0; i < 32; i++) totalSum += finalBuf[i];

  #ifndef HALF_PRINT_ENABLED
  printf("Final Verification Hardware Checksum Trace: %" PRIi64 "\n", totalSum + hiddenZero + liveChecksumAccum);
  #else
  fflush(stdout);
  #endif
  return 0;
}
