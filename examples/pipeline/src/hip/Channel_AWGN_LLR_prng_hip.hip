/**
 * Channel_AWGN_LLR_prng_hip.hip
 * -----------------------------
 * HIP port of Channel_AWGN_LLR_prng_cuda.cu.
 *
 * The generator is include/Rng/Philox4x32.hpp, included unchanged: it uses no vendor intrinsic
 * and no vendor vector type, so the port is exactly what is below -- index arithmetic, the
 * launch, and the counter. There is no rocRAND dependency, mirroring the absence of cuRAND on
 * the CUDA side.
 *
 * Design choices (identical to the CUDA backend, and they have to be: the geometry is part of
 * what makes the two produce the same noise)
 * ----------------------------------------------------------------------------------------
 *  - Counter-based RNG: no state array, so nothing to allocate, initialise, or resize when the
 *    frame count changes, and no shared mutable state for concurrent clones to race on.
 *  - One Philox call yields four 32-bit words -> two Box-Muller pairs -> four noise samples,
 *    so each thread handles four elements and pays the round cost once.
 *  - Those four elements are strided by the total thread count rather than adjacent, which
 *    keeps every memory instruction fully coalesced.
 *  - Works in-place (y can alias x) or out-of-place, and is bounds-checked, so n_samples has
 *    no divisibility constraint.
 */

#include <stdexcept>
#include <cstdio>
#include <hip/hip_runtime.h>

#include "Hip/Channel_AWGN_LLR_prng_hip.hpp"
#include "Rng/Philox4x32.hpp"

#include "Device/Hip/Hip_executor.hpp"

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                    */
/* ------------------------------------------------------------------ */

#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t _e = (call);                                           \
        if (_e != hipSuccess) {                                           \
            fprintf(stderr, "HIP error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_e));           \
            throw std::runtime_error(hipGetErrorString(_e));              \
        }                                                                 \
    } while (0)

static const int THREADS_PER_BLOCK  = 256;
static const int SAMPLES_PER_THREAD = 4; // one Philox4x32 draw feeds exactly four samples

/* ------------------------------------------------------------------ */
/*  Kernel                                                              */
/* ------------------------------------------------------------------ */

/**
 * awgn_add_noise_philox_hip
 * -------------------------
 * Adds AWGN to the input vector:  y[i] = x[i] + sigma * n[i],  n~N(0,1).
 *
 * Thread `idx` owns the elements idx, idx+stride, idx+2*stride, idx+3*stride (stride = total
 * number of threads), so consecutive threads touch consecutive addresses on every access.
 * Its four samples all come from the single Philox draw keyed by (seed, counter, idx).
 */
__global__ void awgn_add_noise_philox_hip(const float* __restrict__ x,
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

sp_hip::Hip_channel_prng::Hip_channel_prng(int device_id)
  : dev_id(device_id), executor(new spu::executor::HIP_executor(device_id)),
    seed_lo(42u), seed_hi(0u), call_counter(0ull)
{
}

void
sp_hip::Hip_channel_prng::set_seed(unsigned long long seed)
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
sp_hip::Hip_channel_prng::add_noise(const float* d_x,
                                    float*       d_y,
                                    int          n_samples,
                                    float        sigma,
                                    spu::device_interface::GpuStream* stream)
{
    if (n_samples <= 0) return;

    auto native_stream = stream->cast<hipStream_t>();

    const int active_threads = (n_samples + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD;
    const int blocks         = (active_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* One counter value per launch, claimed atomically: this is the whole "state" of the
       generator, and it lives on the host, so no device memory is touched between frames. */
    const unsigned long long counter = this->call_counter.fetch_add(1ull);

    // --dec-profile / SPU_LDPC_PROFILE. The events launch_with_profiling() records sit on the same
    // stream as the kernel, so profiling changes what is measured, not where the work runs. No lock:
    // clone() gives every replica of the channel module its own handler, hence its own executor.
    if (!gpu_prof::enabled())
    {
        executor->launch(awgn_add_noise_philox_hip, blocks, THREADS_PER_BLOCK, 0, native_stream,
                         d_x, d_y, n_samples, sigma, this->seed_lo, this->seed_hi,
                         (uint32_t)(counter & 0xFFFFFFFFull), (uint32_t)(counter >> 32));
        executor->synchronize(native_stream);
        return;
    }

    executor->launch_with_profiling(awgn_add_noise_philox_hip, blocks, THREADS_PER_BLOCK, 0, native_stream,
                                    d_x, d_y, n_samples, sigma, this->seed_lo, this->seed_hi,
                                    (uint32_t)(counter & 0xFFFFFFFFull), (uint32_t)(counter >> 32));
    executor->synchronize(native_stream);

    // After the sync: the events only hold a time once the kernel completed. Clearing is mandatory,
    // or the records (and their events) pile up and every call replays the whole history.
    this->prof.add(executor->return_profiling_times());
    executor->clear_profiling_records();
}
