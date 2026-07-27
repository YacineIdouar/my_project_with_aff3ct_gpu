#pragma once

#include <atomic>
#include <cstdint>

#include <streampu.hpp>

namespace sp_vulkan
{

/**
 * AWGN channel handler backed by the hand-written Philox4x32-10 shader
 * (src/vulkan/Shaders/awgn_philox.comp), the Vulkan counterpart of Cuda_channel_prng.
 *
 * Same contract as the CUDA one, and for good reason: both dispatch the same generator with
 * the same index/counter layout, so for identical (n, sigma, seed, counter) the two backends
 * produce bit-identical noise.
 *
 * The generator is counter-based, so this class holds no device state: no buffer to allocate,
 * no initialisation dispatch, and set_seed() is free.
 */
class Vulkan_channel_prng
{
private:
	int dev_id;

	uint32_t seed_lo, seed_hi; // 64-bit seed split into the two Philox key words

	// Incremented once per add_noise() call so successive frames draw disjoint sub-streams.
	// Atomic because clone() shares the handler between module copies, which the pipeline may
	// run concurrently on separate streams.
	std::atomic<unsigned long long> call_counter;

public:
	explicit Vulkan_channel_prng(int device_id);

	// No allocation and no dispatch: just re-keys the generator and rewinds the counter.
	void set_seed(unsigned long long seed);

	void add_noise(const float* d_x,
	               float*       d_y,
	               int          n_samples,
	               float        sigma,
	               spu::device_interface::GpuStream* stream);
};

}
