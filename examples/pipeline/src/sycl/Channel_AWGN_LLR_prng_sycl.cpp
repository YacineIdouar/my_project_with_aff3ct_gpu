/*
SYCL port of Channel_AWGN_LLR_prng_cuda.cu.

The generator is include/Rng/Philox4x32.hpp, included unchanged: it uses no vendor intrinsic and
no vendor vector type, so the port is exactly what is below -- index arithmetic, the submit, and
the counter. Nothing from oneMKL or any other RNG library is involved, mirroring the absence of
cuRAND on the CUDA side.

The nd_range is 1D and deliberately mirrors the CUDA launch: get_global_id(0) plays the role of
CUDA's global thread index and the work-group size matches the CUDA block size, so the strided
element mapping -- and therefore the noise each sample receives -- is identical between the two
backends. (Unlike the decoder there is no 2D block to reproduce here, so no dimension-order
subtlety: see the header comment of Decoder_LDPC_sycl_kernel.cpp for that case.)
*/

#include <sycl/sycl.hpp>
#include <cstdint>

#include "Sycl/Channel_AWGN_LLR_prng_sycl.hpp"
#include "Rng/Philox4x32.hpp"

#include "Device/Sycl/Sycl_executor.hpp"

namespace
{
// Must match the CUDA/HIP launch geometry: the noise a sample gets depends on the total thread
// count, so the backends only agree while these do.
const int THREADS_PER_GROUP  = 256;
const int SAMPLES_PER_THREAD = 4; // one Philox4x32 draw feeds exactly four samples

inline int groups_for(int n, int bs) { return (n + bs - 1) / bs; }
}

sp_sycl::Sycl_channel_prng::Sycl_channel_prng(int device_id, int platform_id)
  : dev_id(device_id), platform_id(platform_id),
    executor(new spu::executor::SYCL_executor(device_id, platform_id)),
    seed_lo(42u), seed_hi(0u), call_counter(0ull)
{
}

void
sp_sycl::Sycl_channel_prng::set_seed(unsigned long long seed)
{
	this->seed_lo = (uint32_t)(seed & 0xFFFFFFFFull);
	this->seed_hi = (uint32_t)(seed >> 32);
	this->call_counter.store(0ull);
}

/**
 * Add AWGN to d_x and write result to d_y (may be the same pointer).
 *
 * y[i] = x[i] + sigma * n[i], n~N(0,1). Work-item `idx` owns the elements idx, idx+stride,
 * idx+2*stride, idx+3*stride (stride = total number of work-items), so neighbouring work-items
 * touch neighbouring addresses; its four samples all come from one Philox draw keyed by
 * (seed, counter, idx). Bounds-checked per element, so n_samples has no divisibility constraint.
 *
 * @param d_x        Device input vector
 * @param d_y        Device output vector (in-place OK)
 * @param n_samples  Number of float samples
 * @param sigma      Noise standard deviation
 * @param stream     StreamPU GPU stream (a sycl::queue) the kernel is submitted on
 */
void
sp_sycl::Sycl_channel_prng::add_noise(const float* d_x,
                                      float*       d_y,
                                      int          n_samples,
                                      float        sigma,
                                      spu::device_interface::GpuStream* stream)
{
	if (n_samples <= 0) return;

	sycl::queue& native_stream = stream->cast<sycl::queue&>();

	const int active_threads = (n_samples + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD;
	const int groups         = groups_for(active_threads, THREADS_PER_GROUP);
	const int global_size    = groups * THREADS_PER_GROUP;

	/* One counter value per submit, claimed atomically: this is the whole "state" of the
	   generator, and it lives on the host, so no device memory is touched between frames. */
	const unsigned long long counter = this->call_counter.fetch_add(1ull);

	const uint32_t seed_lo = this->seed_lo;
	const uint32_t seed_hi = this->seed_hi;
	const uint32_t ctr_lo  = (uint32_t)(counter & 0xFFFFFFFFull);
	const uint32_t ctr_hi  = (uint32_t)(counter >> 32);
	const int      n       = n_samples;

	// Braces, not parentheses: with a single argument each, the parenthesised form is parsed as a
	// function declaration (most vexing parse) rather than a variable.
	sycl::nd_range<1> ndr{ sycl::range<1>{ (size_t)global_size },
	                       sycl::range<1>{ (size_t)THREADS_PER_GROUP } };

	native_stream.submit([&](sycl::handler& h)
	{
		h.parallel_for(ndr, [=](sycl::nd_item<1> item)
		{
			const int idx    = (int)item.get_global_id(0);
			const int stride = (int)item.get_global_range(0);
			if (idx >= n) return;

			float noise[SAMPLES_PER_THREAD];
			philox_awgn_noise4((uint32_t)idx, ctr_lo, ctr_hi, seed_lo, seed_hi, noise);

#pragma unroll
			for (int k = 0; k < SAMPLES_PER_THREAD; k++)
			{
				const int i = idx + k * stride;
				if (i < n) d_y[i] = d_x[i] + sigma * noise[k];
			}
		});
	});

	native_stream.wait();
}
