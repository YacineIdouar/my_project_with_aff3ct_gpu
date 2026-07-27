/**
 * Channel_AWGN_LLR_prng_cuda.cu
 * -----------------------------
 * AWGN channel for a FEC simulator — CUDA backend of the library-free noise generator.
 *
 * Same job as Channel_AWGN_LLR_cuda.cu, but the Gaussian samples come from the hand-written
 * Philox4x32-10 generator in include/Rng/Philox4x32.hpp instead of cuRAND.
 *
 * Everything specific to the generator lives in that header, which depends on nothing but
 * the C++ language: no library, no vendor intrinsic, no vendor vector type. This file holds
 * only the CUDA-shaped part — index arithmetic, the launch, the counter — so a HIP or SYCL
 * port is a transliteration of the ~10 lines below (the Vulkan one has the generator itself
 * ported too, in src/vulkan/Shaders/philox4x32.glsl).
 *
 * Design choices
 * --------------
 *  - Counter-based RNG: no state array, so nothing to allocate, initialise, or resize when
 *    the frame count changes, and no shared mutable state for concurrent clones to race on.
 *  - One Philox call yields four 32-bit words -> two Box-Muller pairs -> four noise samples,
 *    so each thread handles four elements and pays the round cost once.
 *  - Those four elements are strided by the total thread count rather than adjacent, which
 *    keeps every memory instruction fully coalesced.
 *  - The kernel works in-place (y can alias x) or out-of-place, and is bounds-checked, so
 *    n_samples has no divisibility constraint.
 */

#include <stdexcept>
#include <cstdio>

#include "Cuda/Channel_AWGN_LLR_prng_cuda.hpp"
#include "Rng/Philox4x32.hpp"

#include "Device/Cuda/Cuda_executor.hpp"

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                    */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            throw std::runtime_error(cudaGetErrorString(_e));             \
        }                                                                 \
    } while (0)

static const int THREADS_PER_BLOCK  = 256;
static const int SAMPLES_PER_THREAD = 4; // one Philox4x32 draw feeds exactly four samples

/* ------------------------------------------------------------------ */
/*  Kernel                                                              */
/* ------------------------------------------------------------------ */

/**
 * awgn_add_noise_philox
 * ---------------------
 * Adds AWGN to the input vector:  y[i] = x[i] + sigma * n[i],  n~N(0,1).
 *
 * Thread `idx` owns the elements idx, idx+stride, idx+2*stride, idx+3*stride (stride = total
 * number of threads), so consecutive threads touch consecutive addresses on every access.
 * Its four samples all come from the single Philox draw keyed by (seed, counter, idx).
 *
 * @param x        Input samples (device pointer, read-only)
 * @param y        Output samples (device pointer, may equal x)
 * @param n        Number of samples
 * @param sigma    Noise standard deviation
 * @param seed_lo  Low  32 bits of the seed (Philox key word 0)
 * @param seed_hi  High 32 bits of the seed (Philox key word 1)
 * @param ctr_lo   Low  32 bits of the per-call counter
 * @param ctr_hi   High 32 bits of the per-call counter
 */
__global__ void awgn_add_noise_philox(const float* __restrict__ x,
                                      float*       __restrict__ y,
                                      int                       n,
                                      float                     sigma,
                                      uint32_t                  seed_lo,
                                      uint32_t                  seed_hi,
                                      uint32_t                  ctr_lo,
                                      uint32_t                  ctr_hi)
{
    const int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    if (idx >= n) return;

    float noise[SAMPLES_PER_THREAD];
    philox_awgn_noise4((uint32_t)idx, ctr_lo, ctr_hi, seed_lo, seed_hi, noise);

#pragma unroll
    for (int k = 0; k < SAMPLES_PER_THREAD; k++)
    {
        const int i = idx + k * stride;
        if (i < n) y[i] = x[i] + sigma * noise[k];
    }
}

/* ------------------------------------------------------------------ */
/*  Public C++ API                                                      */
/* ------------------------------------------------------------------ */

Cuda_channel_prng::Cuda_channel_prng(int dev_id)
  : dev_id(dev_id), executor(new spu::executor::CUDA_executor(dev_id)),
    seed_lo(42u), seed_hi(0u), call_counter(0ull)
{
}

void
Cuda_channel_prng::set_seed(unsigned long long seed)
{
    this->seed_lo = (uint32_t)(seed & 0xFFFFFFFFull);
    this->seed_hi = (uint32_t)(seed >> 32);
    this->call_counter.store(0ull);
}

/**
 * Add AWGN to d_x and write result to d_y (may be the same pointer).
 *
 * @param d_x        Device input vector
 * @param d_y        Device output vector (in-place OK)
 * @param n_samples  Number of float samples
 * @param sigma      Noise standard deviation
 * @param stream     StreamPU GPU stream the launch is queued on
 */
void
Cuda_channel_prng::add_noise(const float* d_x,
                             float*       d_y,
                             int          n_samples,
                             float        sigma,
                             spu::device_interface::GpuStream* stream)
{
    if (n_samples <= 0) return;

    auto native_stream = stream->cast<cudaStream_t>();

    const int active_threads = (n_samples + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD;
    const int blocks         = (active_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* One counter value per launch, claimed atomically: this is the whole "state" of the
       generator, and it lives on the host, so no device memory is touched between frames. */
    const unsigned long long counter = this->call_counter.fetch_add(1ull);

    executor->launch(awgn_add_noise_philox, blocks, THREADS_PER_BLOCK, 0, native_stream,
                     d_x, d_y, n_samples, sigma, this->seed_lo, this->seed_hi,
                     (uint32_t)(counter & 0xFFFFFFFFull), (uint32_t)(counter >> 32));
    executor->synchronize(native_stream);
}
