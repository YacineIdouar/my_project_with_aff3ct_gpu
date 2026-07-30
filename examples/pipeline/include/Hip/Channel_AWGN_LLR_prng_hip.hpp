#pragma once

#include <atomic>
#include <chrono> // before streampu.hpp on purpose, see below
#include <cstdint>

// StreamPU's Device/Hip/Hip_executor.hpp uses std::chrono (launch_overhead) without including
// <chrono> -- its CUDA equivalent does include it. Pulling <chrono> in first keeps this header
// self-contained instead of relying on whatever the including translation unit happened to pull
// in before it. Remove once the dependency is fixed upstream.
#include <streampu.hpp>

namespace spu { namespace executor { class HIP_executor; } }

namespace sp_hip
{

/**
 * AWGN channel handler backed by the hand-written Philox4x32-10 kernel
 * (src/hip/Channel_AWGN_LLR_prng_hip.hip), the HIP counterpart of Cuda_channel_prng.
 *
 * The generator comes from include/Rng/Philox4x32.hpp, which is shared verbatim with the CUDA
 * and SYCL backends -- so for identical (n, sigma, seed, counter) and the same launch geometry
 * every backend produces bit-identical noise.
 *
 * Counter-based, therefore stateless on the device: no buffer to allocate, no initialisation
 * kernel, and set_seed() is free.
 */
class Hip_channel_prng
{
private:
	int dev_id;
	spu::executor::HIP_executor* executor;

	uint32_t seed_lo, seed_hi; // 64-bit seed split into the two Philox key words

	// Incremented once per add_noise() call so successive frames draw disjoint sub-streams.
	// Atomic because clone() shares the handler between module copies, which the pipeline may
	// run concurrently on separate streams.
	std::atomic<unsigned long long> call_counter;

public:
	explicit Hip_channel_prng(int device_id);

	// No allocation and no kernel launch: just re-keys the generator and rewinds the counter.
	void set_seed(unsigned long long seed);

	void add_noise(const float* d_x,
	               float*       d_y,
	               int          n_samples,
	               float        sigma,
	               spu::device_interface::GpuStream* stream);
};

}
