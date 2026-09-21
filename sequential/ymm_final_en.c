/**
 * @file ymm_final_en.c
 * @brief Single-Threaded AVX2 Permutation Generator for benchmarking
 * 
 * @copyright Copyright (c) 2026 Serge Melnikov. All rights reserved.
 * @license This project is licensed under the MIT License - see the LICENSE file for details.
 * 
 * @details
 * - Core Architecture: Sequential vector engine featuring a production-ready.
 * - Logic: Traverses only the structural half (n!/2) of the permutation space, 
 *   allowing real-time client-side reverse-mirror expansion.
 * 
 * @environment
 * - Platform: Pure Cross-platform compliant (Windows / Linux).
 * - Compiler: GCC / MinGW / MSVC (Best performance achieved via -O3 -mavx2 -funroll-loops).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <inttypes.h>
#include <immintrin.h>

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
    static inline uint64_t tzcnt_u64_fallback(uint64_t value) 
    {
        unsigned long index;
        if (_BitScanForward64(&index, value)) return (uint64_t)index;
        return 64;
    }
    #define TZCNT_U64(x) tzcnt_u64_fallback(x)
#define tzcnt_u64_fallback(x)
#elif defined(__GNUC__) || defined(__clang__)
    /* Builtin CTZ returns undefined for 0, so we must handle the 0 input explicitly */
    static inline uint64_t tzcnt_u64_gcc(uint64_t value) 
    {
        if (value == 0) return 64;
        return (uint64_t)__builtin_ctzll(value);
    }
    #define TZCNT_U64(x) tzcnt_u64_gcc(x)
#else
    static inline uint64_t tzcnt_u64_generic(uint64_t value) 
    {
        if (value == 0) return 64;
        uint64_t count = 0;
        while ((value & 1) == 0) { count++; value >>= 1; }
        return count;
    }
    #define TZCNT_U64(x) tzcnt_u64_generic(x)
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

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <pthread.h>
#endif

void set_thread_affinity() {
#if defined(_WIN32) || defined(_WIN64)
    SetThreadAffinityMask(GetCurrentThread(), 0x01); // Thread affinity binding to Core 0 in Windows
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // Thread affinity binding to Core 0 in Linux
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

typedef int32_t I32;
typedef uint64_t U64;
typedef int64_t I64;
typedef uint32_t U32;
typedef uint16_t U16;
typedef uint8_t U8;

/* Uncomment to enable verbose output */
// #define HALF_PRINT_ENABLED
#define PRINT_TIME 1

ALIGN_32 __m256i ymmMasks[16];
ALIGN_32 __m256i currentYMM;
ALIGN_32 U8 pLanes[32];

/* Global variables for macro support and optimizer bypassing */
I64 n = 0;
I64 hiddenZero = 0;

/* Global register to accumulate ladder vector checksums */
ALIGN_32 __m256i ymmGlobalChecksum = {0};

/* Variable to accumulate the runtime scalar checksum trace */
I64 liveChecksumAccum = 0;

/* Ladder function triggers to print all n elements */
#ifdef HALF_PRINT_ENABLED
  #define PERM_PRINT ProcessExtendedLadder((I32)n, 0);
#else
  #define PERM_PRINT liveChecksumAccum += ProcessExtendedLadder((I32)n, (I32)(totalCount & 0xFF));
#endif

void PrintHalfLanes(const U8* lanes, I32 n)
{
  static U8 chars[16] = "0123456789abcdef";
  const U8* pA = lanes;
  const U8* pB = lanes + 16;
  for (I32 i = 0; i < n; i++) printf("%c", chars[pA[i]]);
  puts("");
  for (I32 i = 0; i < n; i++) printf("%c", chars[pB[i]]);
  puts("");
}

/* Ladder processing function with printouts for all n elements */
static inline I32 ProcessExtendedLadder(I32 n, I32 currentSumSeed)
{
  __m256i localSum = _mm256_set1_epi8((char)currentSumSeed);

  /* 1. Inject element n-1 at the head (1-byte left shift and bitwise OR) */
  __m256i shifted = _mm256_bslli_epi128(currentYMM, 1);
  __m256i n_vector = _mm256_setr_epi8(
      (char)n, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      (char)n, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0
  );
  __m256i ladderYMM = _mm256_or_si256(shifted, n_vector);
  localSum = _mm256_add_epi8(localSum, ladderYMM);

  /* PRINT: Output the initial position of element n-1 in this ladder (length n) */
  #ifdef HALF_PRINT_ENABLED
  ALIGN_32 U8 debugBuf[32];
  _mm256_store_si256((__m256i*)debugBuf, ladderYMM);
  PrintHalfLanes(debugBuf, n + 1); /* Print from debugBuf, which contains element n-1 */
  #endif

  /* 2. Propagate element n through the entire permutation (shuffle steps) */
  UNROLL_16
  for (I64 step = 0; step < n; step++)
   {
     ladderYMM = _mm256_shuffle_epi8(ladderYMM, ymmMasks[step]);
     localSum = _mm256_add_epi8(localSum, ladderYMM);

     /* PRINT: Output each transition step of element n inside the ladder */
     #ifdef HALF_PRINT_ENABLED
     _mm256_store_si256((__m256i*)debugBuf, ladderYMM);
     PrintHalfLanes(debugBuf, n + 1);
     #endif
   }

  /* Update global vector checksum */
  ymmGlobalChecksum = _mm256_add_epi8(ymmGlobalChecksum, ladderYMM);

  ALIGN_32 char extractBuf[32];
  _mm256_store_si256((__m256i*)extractBuf, localSum);

  return (I32)extractBuf[0];
}

void PreparePermYMM_Half(I32 n)
{
  for (I32 i = 0; i < n; i++)
   {
     ALIGN_16 char tempMask[16];
     for (I32 j = 0; j < 16; j++) tempMask[j] = (char)j;
     tempMask[i] = (char)(i + 1);
     tempMask[i + 1] = (char)i;

     __m128i xmmM = _mm_load_si128((const __m128i*)tempMask);
     ymmMasks[i] = _mm256_setr_m128i(xmmM, xmmM);
   }

  /* Lane A: Standard initial state (0, 1, 2, 3...) */
  ALIGN_16 char initA[16] = {0};
  for (I32 i = 0; i < n; i++) initA[i] = (char)i;

  /* Lane B: Invariant setup for the n!/4 sequence 
     (element 3 shifts right: 0, 2, 3, 1, followed by sequential values) */
  ALIGN_16 char initB[16] = {0};
  if (n >= 4)
   {
      initB[0] = 0; initB[1] = 2; initB[2] = 3; initB[3] = 1;
      for (I32 i = 4; i < n; i++) initB[i] = (char)i;
   } else
      for (I32 i = 0; i < n; i++) initB[i] = (char)i;

  currentYMM = _mm256_setr_m128i(_mm_load_si128((const __m128i*)initA), _mm_load_si128((const __m128i*)initB));
}

void GenPermYMM_Half(I32 n)
{
  I64 nminus1 = n - 1, step[2] = {-1, 1}, mask[16], bitPos[16];

  /* --- Lane A Control Variables --- */
  I64 indA[16] = {0};
  I32 dirA = 0, actA = -2;

  /* --- Lane B Control Variables --- */
  I64 indB[16] = {0};
  I32 dirB = 1 << (n - 4), actB = 0xfffffffe;

  /* Extract initial element positions from the startup YMM register for the inverse lookups */
  ALIGN_32 char initBuf[32];
  _mm256_store_si256((__m256i*)initBuf, currentYMM);

  for (I32 i = 0; i < n; i++) {
      indA[(U8)initBuf[i]] = i;
      indB[(U8)initBuf[i + 16]] = i;
  }

  /* Compute and populate baseline masks */
  for (I32 i = 0; i < n; i++) {
      bitPos[i] = 1ULL << (nminus1 - i);
      mask[i] = bitPos[i] - 1ULL;
  }

  I64 targetPerm = 1;
  for (I64 i = 1; i <= n; i++) targetPerm *= i;
  targetPerm /= 4;
  I64 totalCount = 0;

  PERM_PRINT;
  totalCount++;

  while (1)
   { /* --- LEFTWARD SWEEP --- */
     for (I32 i = nminus1; i > 0; i--)
     {
       currentYMM = _mm256_shuffle_epi8(currentYMM, ymmMasks[i - 1]);
       PERM_PRINT;
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
     I32 maskIdxA = curIndA < targetIndA ? curIndA : targetIndA;

     /* --- INTERSECTION 1: LANE B --- */
     I64 curIndB = indB[curItemB] + 1;
     I64 curDirB = step[(dirB >> (nminus1 - curItemB)) & 1];
     I64 targetIndB = curIndB + curDirB;
     I32 maskIdxB = curIndB < targetIndB ? curIndB : targetIndB;

     __m128i lowTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 0), _mm256_castsi256_si128(ymmMasks[maskIdxA]));
     __m128i highTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 1), _mm256_castsi256_si128(ymmMasks[maskIdxB]));
     currentYMM = _mm256_setr_m128i(lowTransformed, highTransformed);

     PERM_PRINT;
     totalCount++;

     I64 itemDisplacedA = pA[targetIndA];
     indA[curItemA] = targetIndA - 1; indA[itemDisplacedA] = curIndA - 1;

     I64 itemDisplacedB = pB[targetIndB];
     indB[curItemB] = targetIndB - 1; indB[itemDisplacedB] = curIndB - 1;

     curIndA += curDirA;
     dirA ^= mask[curItemA];
     actA |= mask[curItemA]; actA ^= 1;
     I64 isBlockedA = 1;
     I64 nextA = curIndA + curDirA;
     if (nextA >= 1 && nextA <= nminus1 && curItemA > pA[nextA]) isBlockedA = 0;
     if (isBlockedA) actA &= ~bitPos[curItemA];

     curIndB += curDirB;
     dirB ^= mask[curItemB];
     actB |= mask[curItemB]; actB ^= 1;
     I64 isBlockedB = 1;
     I64 nextB = curIndB + curDirB;
     if (nextB >= 1 && nextB <= nminus1 && curItemB > pB[nextB]) isBlockedB = 0;
     if (isBlockedB) actB &= ~bitPos[curItemB];

     /* --- RIGHTWARD SWEEP --- */
     for (I32 i = 0; i < nminus1; i++)
     {
       currentYMM = _mm256_shuffle_epi8(currentYMM, ymmMasks[i]);
       PERM_PRINT;
     }
     totalCount += nminus1;
     if (totalCount >= targetPerm) return;

     _mm256_store_si256((__m256i*)pLanes, currentYMM);

     curItemA = nminus1 - TZCNT_U64(actA);
     curItemB = nminus1 - TZCNT_U64(actB);
     // OR "if (curItemB == 1) return;" INSTEAD OF "if (totalCount >= targetPerm) return;" ABOVE

     /* --- INTERSECTION 2: LANE A --- */
     curIndA = indA[curItemA];
     curDirA = step[(dirA >> (nminus1 - curItemA)) & 1];
     targetIndA = curIndA + curDirA;
     maskIdxA = curIndA < targetIndA ? curIndA : targetIndA;

     /* --- INTERSECTION 2: LANE B --- */
     curIndB = indB[curItemB];
     curDirB = step[(dirB >> (nminus1 - curItemB)) & 1];
     targetIndB = curIndB + curDirB;
     maskIdxB = curIndB < targetIndB ? curIndB : targetIndB;

     lowTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 0), _mm256_castsi256_si128(ymmMasks[maskIdxA]));
     highTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 1), _mm256_castsi256_si128(ymmMasks[maskIdxB]));
     currentYMM = _mm256_setr_m128i(lowTransformed, highTransformed);

     PERM_PRINT;
     totalCount++;

     itemDisplacedA = pA[targetIndA];
     indA[curItemA] = targetIndA; indA[itemDisplacedA] = curIndA;

     itemDisplacedB = pB[targetIndB];
     indB[curItemB] = targetIndB; indB[itemDisplacedB] = curIndB;

     curIndA += curDirA;
     dirA ^= mask[curItemA];
     actA |= mask[curItemA]; actA ^= 1;
     isBlockedA = 1; nextA = curIndA + curDirA;
     if (nextA >= 0 && nextA <= nminus1 && curItemA > pA[nextA]) isBlockedA = 0;
     if (isBlockedA) actA &= ~bitPos[curItemA];

     curIndB += curDirB;
     dirB ^= mask[curItemB];
     actB |= mask[curItemB]; actB ^= 1;
     isBlockedB = 1;
     nextB = curIndB + curDirB;
     if (nextB >= 0 && nextB <= nminus1 && curItemB > pB[nextB]) isBlockedB = 0;
     if (isBlockedB) actB &= ~bitPos[curItemB];
  }
}

int main(int argc, char* argv[])
{ 
  U64 count1, count2;
  U32 cpuInfo[4];

  if (argc != 2)
  {
    fprintf(stderr, "Usage: %s <n>\n", argv[0]);
    return 1;
  }
  n = atoi(argv[1]);
  if (n < 6 || n > 16)
  {
    fputs("Error: wrong parameter (must be 6..16)\n", stderr);
    return 1;
  }
  --n;
  hiddenZero = argc / 100;

  set_thread_affinity(); // optional

  // CPU warmup (forcing Turbo Boost frequency to its maximum)
  volatile int warm_dummy = 42;
  for (int i = 0; i < 50000000; ++i) warm_dummy = warm_dummy * 3 + i;

  // Serialize and record benchmark start time
#ifdef _MSC_VER
  __cpuid(cpuInfo, 0);
#else
  __get_cpuid(0, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
#endif
  _mm_lfence();
  count1 = __rdtsc();
  _mm_lfence();

  PreparePermYMM_Half((I32)n);
  GenPermYMM_Half((I32)n);

  // Serialize and record benchmark end time
  _mm_lfence();
  count2 = __rdtsc();
  _mm_lfence();

  if (PRINT_TIME == 1)
   {
     #ifndef HALF_PRINT_ENABLED
     printf("%" PRIu64 " CPU cycles\n", count2 - count1);
     #endif
   }
  
  /* Compiler barrier to protect the global vector checksum from optimization bypassing */
  COMPILER_BARRIER();

  ALIGN_32 char finalBuf[32];
  _mm256_store_si256((__m256i*)finalBuf, ymmGlobalChecksum);
  I64 totalSum = 0;
  for(I32 i = 0; i < 32; i++) totalSum += finalBuf[i];

  /* Output checksum only in production mode (keeps stdout clean when HALF_PRINT_ENABLED is active) */
  #ifndef HALF_PRINT_ENABLED
  printf("Final Universal Checksum: %" PRIi64 "\n", liveChecksumAccum + totalSum + hiddenZero);
  #else
  /* Flush stdout in debug mode to ensure all generated permutations are flushed to a file/console */
  fflush(stdout);
  #endif
  return 0;
}
