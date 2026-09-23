#pragma once

#include <atomic>
#include <chrono> // before streampu.hpp on purpose, see below
#include <cstdint>

// StreamPU's Device/Hip/Hip_executor.hpp uses std::chrono (launch_overhead) without including
// <chrono> -- its CUDA equivalent does include it. Pulling <chrono> in first keeps this header
// self-contained instead of relying on whatever the including translation unit happened to pull
// in before it. Remove once the dependency is fixed upstream.
#include <streampu.hpp>
#include "Module/Decoder_gpu/gpu_decoder_profiling.hpp"

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
 * Counter-based, so no RNG state lives on the device and set_seed() is free. The one device
 * allocation is two words holding the per-frame counter: passing the counter as a kernel argument
 * instead would change the argument bytes every frame, and those bytes are the key StreamPU caches
 * a captured HIP graph under -- every frame would miss, capture and keep a new graph. Read through
 * a fixed pointer, the arguments are constant and the graph is captured once.
 */
class Hip_channel_prng
{
private:
	int dev_id;
	spu::executor::HIP_executor* executor;
	// Per-handler profiling totals, registered process-wide; see
	// Module/Decoder_gpu/gpu_decoder_profiling.hpp.
	gpu_prof::accumulator prof{ "CHANNEL", "HIP" };


	uint32_t seed_lo, seed_hi; // 64-bit seed split into the two Philox key words

	// Incremented once per add_noise() call so successive frames draw disjoint sub-streams.
	// Atomic because clone() shares the handler between module copies, which the pipeline may
	// run concurrently on separate streams.
	std::atomic<unsigned long long> call_counter;

	// The counter the kernel reads, as two words: 'd_ctr' on the device, passed to every launch as
	// the same pointer, and 'h_ctr' a pinned host staging copy it is refreshed from before each
	// launch. Pinned so the asynchronous copy reads stable memory. Allocated on the first add_noise(),
	// on dev_id.
	uint32_t* d_ctr = nullptr;
	uint32_t* h_ctr = nullptr;

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
