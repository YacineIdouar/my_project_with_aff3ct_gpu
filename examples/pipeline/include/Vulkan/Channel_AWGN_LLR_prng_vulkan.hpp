#pragma once

#include <atomic>
#include <cstdint>

#include <streampu.hpp>
#include "Module/Decoder_gpu/gpu_decoder_profiling.hpp"

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
// Everything the dispatch is described with, kept alive between calls: the two socket VkBuffers,
// the counter buffer and its host mapping, the push constants the descriptors point at, and the
// one-entry chain itself. Defined in the .cpp -- naming it here would drag vulkan.h into every
// translation unit that includes this header, obj-module's among them.
struct Awgn_dispatch_cache;

class Vulkan_channel_prng
{
private:
	int dev_id;
	// Per-handler profiling totals, registered process-wide; see
	// Module/Decoder_gpu/gpu_decoder_profiling.hpp.
	gpu_prof::accumulator prof{ "CHANNEL", "VULKAN" };

	uint32_t seed_lo, seed_hi; // 64-bit seed split into the two Philox key words

	// Incremented once per add_noise() call so successive frames draw disjoint sub-streams.
	// Left atomic although Channel_AWGN_LLR_prng_gpu::clone() now gives each replica its own
	// handler: it costs nothing on an uncontended counter, and it is the one member that would
	// silently produce correlated noise if a handler were ever shared again.
	std::atomic<unsigned long long> call_counter;

	// Built on the first add_noise() and reused afterwards; freed by the destructor. Held by
	// pointer so this header stays free of Vulkan types.
	Awgn_dispatch_cache* cache = nullptr;

public:
	explicit Vulkan_channel_prng(int device_id);
	~Vulkan_channel_prng();

	Vulkan_channel_prng(const Vulkan_channel_prng&) = delete;
	Vulkan_channel_prng& operator=(const Vulkan_channel_prng&) = delete;

	// No allocation and no dispatch: just re-keys the generator and rewinds the counter.
	void set_seed(unsigned long long seed);

	void add_noise(const float* d_x,
	               float*       d_y,
	               int          n_samples,
	               float        sigma,
	               spu::device_interface::GpuStream* stream);
};

}
