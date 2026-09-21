/**
 * @file ymm_final1_en.c
 * @brief Practical Single-Threaded AVX2 Permutation Generator with Callback
 * 
 * @copyright Copyright (c) 2026 Serge Melnikov. All rights reserved.
 * @license This project is licensed under the MIT License - see the LICENSE file for details.
 * 
 * @details
 * - Core Architecture: Sequential vector engine featuring a production-ready, 
 *   high-speed user callback interface (PermCallback) executed directly out of registers.
 * - Logic: Traverses only the structural half (n!/2) of the permutation space, 
 *   allowing real-time client-side reverse-mirror expansion.
 * 
 * @environment
 * - Platform: Pure Cross-platform compliant (Windows / Linux).
 * - Compiler: GCC / MinGW / MSVC (Best performance achieved via -O3 -mavx2 -funroll-loops).
 */

#include <stdio.h>
#include <inttypes.h>
#include <immintrin.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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

/* Cross-compiler trailing zero count configuration (safe for non-zero inputs) */
#if defined(_MSC_VER)
    #include <intrin.h>
    static inline uint64_t tzcnt_u64_fast(uint64_t value) {
        unsigned long index;
        _BitScanForward64(&index, value);
        return (uint64_t)index;
    }
    #define TZCNT_U64(x) tzcnt_u64_fast(x)
#elif defined(__GNUC__) || defined(__clang__)
    /* Generates a single optimal instruction without any branching or checks */
    #define TZCNT_U64(x) ((uint64_t)__builtin_ctzll(x))
#else
    #define TZCNT_U64(x) 0
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

/* Uncomment to enable user callback operations (e.g., verbose output) */
// #define HALF_PRINT_ENABLED
#define PRINT_TIME 1

/* User-defined callback function pointer type for processing permutations */
typedef void (*PermCallback)(const U8* lanes, I32 n);

ALIGN_32 __m256i ymmMasks[16];
ALIGN_32 __m256i currentYMM;
ALIGN_32 U8 pLanes[32];

I64 n = 0;
I64 hiddenZero = 0;
ALIGN_32 __m256i ymmGglobalChecksum = {0};
I64 liveChecksumAccum = 0;

/* Demonstration user callback that prints the generated permutations */
void MyDemoCallback(const U8* singleLane, I32 length)
{
  static U8 chars[] = "0123456789abcdef";

  /* 1. Print the permutation in forward order (left-to-right) */
  for (I32 i = 0; i < length; i++) printf("%c", chars[singleLane[i]]);
  puts("");

  /* 2. Print the same permutation in reverse order (right-to-left).
     This operation effectively yields the remaining half of the n! permutations 
     at zero additional computational cost. */
  for (I32 i = length - 1; i >= 0; i--) printf("%c", chars[singleLane[i]]);
  puts("");
}

static inline I32 ProcessExtendedLadder(I32 n, I32 currentSumSeed, PermCallback cb)
{
  __m256i localSum = _mm256_set1_epi8((char)currentSumSeed);

  __m256i shifted = _mm256_bslli_epi128(currentYMM, 1);
  __m256i nVector = _mm256_setr_epi8(
      (char)n, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      (char)n, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0
  );
  __m256i ladderYMM = _mm256_or_si256(shifted, nVector);
  localSum = _mm256_add_epi8(localSum, ladderYMM);

  #ifdef HALF_PRINT_ENABLED
  if (cb)
   {
     ALIGN_32 U8 debugBuf[32];
     _mm256_store_si256((__m256i*)debugBuf, ladderYMM);
     cb(debugBuf, n + 1);      /* Lane A mapped to the continuous output column */
     cb(debugBuf + 16, n + 1); /* Lane B mapped to the continuous output column */
   }
  #endif

  UNROLL_16
  for (I64 step = 0; step < n; step++)
   {
     ladderYMM = _mm256_shuffle_epi8(ladderYMM, ymmMasks[step]);
     localSum = _mm256_add_epi8(localSum, ladderYMM);

     #ifdef HALF_PRINT_ENABLED
     if (cb)
      {
        ALIGN_32 U8 debugBuf[32];
        _mm256_store_si256((__m256i*)debugBuf, ladderYMM);
        cb(debugBuf, n + 1);
        cb(debugBuf + 16, n + 1);
      }
     #endif
   }

  ymmGglobalChecksum = _mm256_add_epi8(ymmGglobalChecksum, ladderYMM);

  ALIGN_32 char extractBuf[32];
  _mm256_store_si256((__m256i*)extractBuf, localSum);

  return (I32)extractBuf[0];
}

#ifdef HALF_PRINT_ENABLED
  #define PERM_PRINT ProcessExtendedLadder((I32)n, 0, cb);
#else
  #define PERM_PRINT liveChecksumAccum += ProcessExtendedLadder((I32)n, (I32)(totalCount & 0xFF), NULL);
#endif

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

  ALIGN_16 char initA[16] = {0};
  for (I32 i = 0; i < n; i++) initA[i] = (char)i;

  ALIGN_16 char initB[16] = {0};
  if (n >= 4)
   {
     initB[0] = 0; initB[1] = 2; initB[2] = 3; initB[3] = 1;
     for (I32 i = 4; i < n; i++) initB[i] = (char)i;
   }
  else
   for (I32 i = 0; i < n; i++) initB[i] = (char)i;

  currentYMM = _mm256_setr_m128i(_mm_load_si128((const __m128i*)initA), _mm_load_si128((const __m128i*)initB));
}

void GenPermYMM_Half(I32 n, PermCallback cb)
{
  I64 nminus1 = n - 1, step[2] = {-1, 1}, mask[16], bitPos[16];

  I64 indA[16] = {0};
  I32 dirA = 0, actA = -2;

  I64 indB[16] = {0};
  I32 dirB = 1 << (n - 4), actB = 0xfffffffe;

  /* Extract initial tracking states into aligned memory */
  ALIGN_32 char initBuf[32];
  _mm256_store_si256((__m256i*)initBuf, currentYMM);

  for (I32 i = 0; i < n; i++)
   {
     indA[(unsigned char)initBuf[i]] = i;
     indB[(unsigned char)initBuf[i + 16]] = i;
   }

  for (I32 i = 0; i < n; i++)
   {
     bitPos[i] = 1ULL << (nminus1 - i);
     mask[i] = bitPos[i] - 1ULL;
   }

  I64 target_permutations = 1;
  for (I64 i = 1; i <= n; i++) target_permutations *= i;
  target_permutations /= 4;
  I64 totalCount = 0;

  PERM_PRINT;
  totalCount++;

  while (1)
   {
     /* --- LEFTWARD SWEEP --- */
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
     I64 currindA = indA[curItemA] + 1;
     I64 currdirA = step[(dirA >> (nminus1 - curItemA)) & 1];
     I64 targetIndA = currindA + currdirA;
     I32 maskIdxA = currindA < targetIndA ? currindA : targetIndA;

     /* --- INTERSECTION 1: LANE B --- */
     I64 currindB = indB[curItemB] + 1;
     I64 currdirB = step[(dirB >> (nminus1 - curItemB)) & 1];
     I64 targetIndB = currindB + currdirB;
     I32 maskIdxB = currindB < targetIndB ? currindB : targetIndB;

     __m128i lowTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 0), _mm256_castsi256_si128(ymmMasks[maskIdxA]));
     __m128i highTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 1), _mm256_castsi256_si128(ymmMasks[maskIdxB]));
     currentYMM = _mm256_setr_m128i(lowTransformed, highTransformed);

     PERM_PRINT;
     totalCount++;

     I64 itemDisplacedA = pA[targetIndA];
     indA[curItemA] = targetIndA - 1; indA[itemDisplacedA] = currindA - 1;

     I64 itemDisplacedB = pB[targetIndB];
     indB[curItemB] = targetIndB - 1; indB[itemDisplacedB] = currindB - 1;

     currindA += currdirA;
     dirA ^= mask[curItemA];
     actA |= mask[curItemA]; actA ^= 1;
     I64 isBlockedA = 1;
     I64 nextA = currindA + currdirA;
     if (nextA >= 1 && nextA <= nminus1 && curItemA > pA[nextA]) isBlockedA = 0;
     if (isBlockedA) actA &= ~bitPos[curItemA];

     currindB += currdirB;
     dirB ^= mask[curItemB];
     actB |= mask[curItemB]; actB ^= 1;
     I64 isBlockedB = 1;
     I64 nextB = currindB + currdirB;
     if (nextB >= 1 && nextB <= nminus1 && curItemB > pB[nextB]) isBlockedB = 0;
     if (isBlockedB) actB &= ~bitPos[curItemB];

     /* --- RIGHTWARD SWEEP --- */
     for (I32 i = 0; i < nminus1; i++)
      {
        currentYMM = _mm256_shuffle_epi8(currentYMM, ymmMasks[i]);
        PERM_PRINT;
      }
     totalCount += nminus1;
     if (totalCount >= target_permutations) return;

     _mm256_store_si256((__m256i*)pLanes, currentYMM);

     curItemA = nminus1 - TZCNT_U64(actA);
     curItemB = nminus1 - TZCNT_U64(actB);
     // OR "if (curItemB == 1) return;" INSTEAD OF "if (totalCount >= targetPerm) return;" ABOVE

     /* --- INTERSECTION 2: LANE A --- */
     currindA = indA[curItemA];
     currdirA = step[(dirA >> (nminus1 - curItemA)) & 1];
     targetIndA = currindA + currdirA;
     maskIdxA = currindA < targetIndA ? currindA : targetIndA;

     /* --- INTERSECTION 2: LANE B --- */
     currindB = indB[curItemB];
     currdirB = step[(dirB >> (nminus1 - curItemB)) & 1];
     targetIndB = currindB + currdirB;
     maskIdxB = currindB < targetIndB ? currindB : targetIndB;

     lowTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 0), _mm256_castsi256_si128(ymmMasks[maskIdxA]));
     highTransformed = _mm_shuffle_epi8(_mm256_extracti128_si256(currentYMM, 1), _mm256_castsi256_si128(ymmMasks[maskIdxB]));
     currentYMM = _mm256_setr_m128i(lowTransformed, highTransformed);

     PERM_PRINT;
     totalCount++;

     itemDisplacedA = pA[targetIndA];
     indA[curItemA] = targetIndA; indA[itemDisplacedA] = currindA;

     itemDisplacedB = pB[targetIndB];
     indB[curItemB] = targetIndB; indB[itemDisplacedB] = currindB;

     currindA += currdirA;
     dirA ^= mask[curItemA];
     actA |= mask[curItemA]; actA ^= 1;
     isBlockedA = 1;
     nextA = currindA + currdirA;
     if (nextA >= 0 && nextA <= nminus1 && curItemA > pA[nextA]) isBlockedA = 0;
     if (isBlockedA) actA &= ~bitPos[curItemA];

     currindB += currdirB;
     dirB ^= mask[curItemB];
     actB |= mask[curItemB]; actB ^= 1;
     isBlockedB = 1;
     nextB = currindB + currdirB;
     if (nextB >= 0 && nextB <= nminus1 && curItemB > pB[nextB]) isBlockedB = 0;
     if (isBlockedB) actB &= ~bitPos[curItemB];
   }
}

/* Cross-compiler optimization barrier configuration */
#if defined(_MSC_VER)
    #include <intrin.h>
    #pragma intrinsic(_ReadWriteBarrier)
    #define COMPILER_BARRIER() _ReadWriteBarrier()
#elif defined(__GNUC__) || defined(__clang__)
    #define COMPILER_BARRIER() __asm__ volatile("" : : "g" (&ymmGglobalChecksum) : "memory")
#else
    #define COMPILER_BARRIER()
#endif

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
  #ifdef HALF_PRINT_ENABLED
  GenPermYMM_Half((I32)n, MyDemoCallback);
  #else
  GenPermYMM_Half((I32)n, NULL);
  #endif

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
  _mm256_store_si256((__m256i*)finalBuf, ymmGglobalChecksum);

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
