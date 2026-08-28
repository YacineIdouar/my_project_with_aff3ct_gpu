/**
 * awgn_channel_cuda.cu
 * --------------------
 * AWGN channel for a FEC simulator — CUDA implementation.
 *
 * Design choices
 * --------------
 *  - RNG state is allocated once per simulation run and reused across
 *    Monte-Carlo frames to avoid re-initialisation overhead.
 *  - curand_normal2() returns two samples per call (Box-Muller pairs),
 *    improving throughput on memory-bound workloads.
 *  - The kernel works in-place (y can alias x) or out-of-place.
 *  - Thread count is padded to even numbers so every thread can consume
 *    a full float2 pair without bounds-check overhead in the hot path.
 */

#include <stdexcept>
#include <cstdio>
//#include <cmath>
#include <iostream>

#include "Cuda/Channel_AWGN_LLR_cuda.hpp"

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                    */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            throw std::runtime_error(cudaGetErrorString(_e));            \
        }                                                                 \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Kernels                                                             */
/* ------------------------------------------------------------------ */

// Global variable to deal with thread states



/**
 * init_rng_states
 * ---------------
 * One-time initialisation of one curandStateXORWOW per thread.
 * Each thread gets a unique sequence via its global index so streams
 * are statistically independent.
 */
__global__ void init_rng_states(curandStateXORWOW_t* __restrict__ states,
                                 int                               n_states,
                                 unsigned long long                seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    /* sequence = idx ensures independence; offset = 0 */
    curand_init(seed, (unsigned long long)idx, 0ULL, &states[idx]);
}

/**
 * awgn_add_noise
 * --------------
 * Adds AWGN to the input vector:  y[i] = x[i] + sigma * n[i],  n~N(0,1).
 *
 * We process two elements per thread using curand_normal2() to exploit
 * Box-Muller pairing.  n must therefore be even (pad on the host side).
 *
 * @param x       Input samples (device pointer, read-only)
 * @param y       Output samples (device pointer, may equal x)
 * @param n       Number of samples (must be even)
 * @param sigma   Noise standard deviation
 * @param states  Persistent RNG states (one per thread)
 */
__global__ void awgn_add_noise(const float* __restrict__ x,
                                float*       __restrict__ y,
                                int                       n,
                                float                     sigma,
                                curandStateXORWOW_t*      states)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 2;          /* each thread owns two samples */
    if (base >= n) return;

    /* Load local copy of RNG state for speed (avoids repeated global loads) */
    curandStateXORWOW_t local_state = states[idx];

    float2 noise = curand_normal2(&local_state); /* paired Box-Muller */

    y[base]     = x[base]     + sigma * noise.x;
    if (base + 1 < n)
        y[base + 1] = x[base + 1] + sigma * noise.y;

    /* Write state back so next call continues the same stream */
    states[idx] = local_state;
}

/* ------------------------------------------------------------------ */
/*  Public C++ API                                                      */
/* ------------------------------------------------------------------ */
/**
 * Allocate and initialise RNG states for up to max_samples samples.
 * Call once per simulation, then reuse across all frames / SNR points.
 */

Cuda_channel::Cuda_channel(int dev_id) : dev_id(dev_id), executor(new spu::executor::CUDA_executor(dev_id)), d_states(nullptr)
{
}

void
Cuda_channel::init_rand_state(int max_samples, int seed, int threads)
{
	cudaSetDevice(dev_id);
    /* We need ceil(max_samples/2) threads */
    int n_states = (max_samples + 1) / 2;
    CUDA_CHECK(cudaMalloc(&d_states,
                          (size_t)n_states * sizeof(curandStateXORWOW_t)));
    int blocks = (n_states + threads - 1) / threads;
    init_rng_states<<<blocks, threads>>>(static_cast<curandStateXORWOW_t*>(d_states), n_states, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Add AWGN to d_x and write result to d_y (may be the same pointer).
 *
 * @param d_x        Device input vector
 * @param d_y        Device output vector (in-place OK)
 * @param n_samples  Number of float samples
 * @param sigma      Noise standard deviation (use awgn_sigma_from_EbN0)
 * @param stream     CUDA stream (default: 0)
 */
void 
Cuda_channel::add_noise(const float*  d_x,
               float*        d_y,
               int           n_samples,
               float         sigma,
			   spu::device_interface::GpuStream* stream)
{
	auto native_stream = stream->cast<cudaStream_t>();
    int active_threads = (n_samples + 1) / 2;
    int blocks = (active_threads + 256 - 1) / 256;

	// --dec-profile / SPU_LDPC_PROFILE. The events launch_with_profiling() records sit on the same
	// stream as the kernel, so profiling changes what is measured, not where the work runs. No lock:
	// clone() gives every replica of the channel module its own handler, hence its own executor.
	if (!gpu_prof::enabled())
	{
		executor->launch(awgn_add_noise, blocks, 256, 0, native_stream, d_x, d_y, n_samples, sigma,
		                 static_cast<curandStateXORWOW_t*>(d_states));
		executor->synchronize(native_stream);
		return;
	}

	executor->launch_with_profiling(
	  awgn_add_noise, blocks, 256, 0, native_stream, d_x, d_y, n_samples, sigma,
	  static_cast<curandStateXORWOW_t*>(d_states));
	executor->synchronize(native_stream);

	// After the sync: the events only hold a time once the kernel completed. Clearing is mandatory,
	// or the records (and their events) pile up and every call replays the whole history.
	this->prof.add(executor->return_profiling_times());
	executor->clear_profiling_records();
}

