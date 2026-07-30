#pragma once

#include <atomic>
#include <cstdint>

#include <streampu.hpp>

namespace spu { namespace executor { class SYCL_executor; } }

namespace sp_sycl
{

/**
 * AWGN channel handler backed by the hand-written Philox4x32-10 kernel
 * (src/sycl/Channel_AWGN_LLR_prng_sycl.cpp), the SYCL counterpart of Cuda_channel_prng.
 *
 * The generator comes from include/Rng/Philox4x32.hpp, which is shared verbatim with the CUDA
 * and HIP backends -- so for identical (n, sigma, seed, counter) and the same launch geometry
 * every backend produces bit-identical noise. No oneMKL / oneAPI RNG dependency, mirroring the
 * absence of cuRAND on the CUDA side.
 *
 * Counter-based, therefore stateless on the device: no buffer to allocate, no initialisation
 * kernel, and set_seed() is free.
 */
class Sycl_channel_prng
{
private:
	int dev_id;
	int platform_id;
	spu::executor::SYCL_executor* executor;

	uint32_t seed_lo, seed_hi; // 64-bit seed split into the two Philox key words

	// Incremented once per add_noise() call so successive frames draw disjoint sub-streams.
	// Atomic because clone() shares the handler between module copies, which the pipeline may
	// run concurrently on separate queues.
	std::atomic<unsigned long long> call_counter;

public:
	explicit Sycl_channel_prng(int device_id, int platform_id = 0);

	// No allocation and no kernel launch: just re-keys the generator and rewinds the counter.
	void set_seed(unsigned long long seed);

	void add_noise(const float* d_x,
	               float*       d_y,
	               int          n_samples,
	               float        sigma,
	               spu::device_interface::GpuStream* stream);
};

}
