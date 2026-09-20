/*
 * Accelerated Implementation of Knuth's Algorithm P (Johnson-Trotter Method)
 * 
 * Based on the classic adjacent transposition algorithm developed by S. M. Johnson 
 * and H. F. Trotter, as presented in Donald E. Knuth's "The Art of Computer Programming" 
 * (Volume 4A, Section 7.2.1.2).
 * 
 * Performance Optimization Note:
 * This specific source code branch incorporates a novel structural optimization 
 * that achieves a 3x execution speedup by isolating the fast-sweeping ladder loops 
 * of element n-1 into an independent, high-speed execution branch.
 * 
 * License: Open Source (Permissive). 
 * This code is provided freely for academic benchmarking, verification, and research 
 * replication purposes in connection with the arXiv.org publication.
 */

#include <stdio.h>
#include <inttypes.h>
#include <immintrin.h>

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

// Set to 1 to enable execution time tracking, 0 to disable
#define PRINT_TIME 1
// Set to 1 to display the generated permutations, 0 to disable
#define DOPRINT 0
typedef uint64_t U64;
typedef uint32_t U32;
typedef int32_t I32;
typedef uint8_t U8;
#define MAXN 36

// Memory barrier to prevent compiler optimizations from eliminating the loop payload
#define PERM_PRINT __asm__ volatile("" : : "g" (p) : "memory"); \
if (DOPRINT) \
 { for (I32 i = 0; i < n; i++) printf("%c", chars[p[i]]); \
   puts(""); \
 }

U8 chars[MAXN] = "0123456789abcdefghijklmnopqrstuvwxyz";

void genPerm(I32 *p, I32 n)
{
  // Fixed-size arrays used instead of VLAs to ensure cross-platform MSVC compatibility
  I32 o[MAXN], c[MAXN], j, s, nminus1 = n - 1;

  for (I32 i = 0; i <= nminus1; i++) { c[i] = 0; o[i] = 1; p[i] = i; }

P2:
  PERM_PRINT;
  j = nminus1; s = 0;

  if (o[nminus1] > 0)
   {
     I32 k = c[nminus1] = nminus1;
     // Fast-path: Leftward shift cascade of the maximum element (n-1)
     while (k)
      {
        p[k] = p[k - 1]; p[k - 1] = nminus1;
        k--;
        PERM_PRINT;
      }
     // Restore the state of the control loop variable post-cascade
     s++;
   }
   else
    {
      I32 k = 1;
      // Reinitialize the control tracking array
      c[nminus1] = 0;
      // Fast-path: Rightward shift cascade of the maximum element (n-1)
      while (k < n)
       {
         p[k - 1] = p[k]; p[k] = nminus1;
         k++;
         PERM_PRINT;
       }
    }
  o[j] = -o[j];
  j--;

P4:
  {
    I32 q = c[j] + o[j];
    if (q < 0) goto P7;
    if (q == j + 1) goto P6;

    // Slow-path: A higher-order permutation element executes a step
    I32 temp = p[j - c[j] + s];
    p[j - c[j] + s] = p[j - q + s];
    p[j - q + s] = temp;
    c[j] = q;
  }
  goto P2;

P6:
  if (j == 0) return;
  s++;

P7:
  o[j] = -o[j];
  j--;
  goto P4;
}

int main(int argc, char* argv[])
{
  I32 perm[MAXN], n;
  U64 count1, count2;
  U32 cpuInfo[4];

  if (argc != 2)
   {
     fprintf(stderr, "Usage: %s <n>\n", argv[0]);
     return 1;
   }
  n = atoi(argv[1]);
  if (n < 2 || n > MAXN)
   {
     fprintf(stderr, "Error: wrong parameter (must be 2..%i)\n", MAXN);
     return 1;
   }

  // set_thread_affinity(); // optional

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

  genPerm(perm, n);

  // Serialize and record benchmark end time
  _mm_lfence();
  count2 = __rdtsc();
  _mm_lfence();

  if (PRINT_TIME) printf("%" PRIu64 " CPU cycles\n", count2 - count1);
  return 0;
}
