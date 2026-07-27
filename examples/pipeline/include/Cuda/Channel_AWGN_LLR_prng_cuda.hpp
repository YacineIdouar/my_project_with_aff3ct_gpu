#ifndef Channel_AWGN_LLR_PRNG_CUDA_HPP_
#define Channel_AWGN_LLR_PRNG_CUDA_HPP_

#include <atomic>
#include <cstdint>

#include <streampu.hpp>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace spu { namespace executor { class CUDA_executor; } }

/**
 * AWGN channel handler backed by a hand-written Philox4x32-10 kernel (see Philox4x32.cuh),
 * the cuRAND-free counterpart of Cuda_channel.
 *
 * The generator is counter-based, so unlike Cuda_channel this class holds no device state:
 * there is no d_states buffer, no initialisation kernel, and set_seed() is free. The noise
 * for a given sample is a pure function of (seed, call counter, thread index).
 */
class Cuda_channel_prng
{
private:
	int dev_id;
	spu::executor::CUDA_executor* executor;

	unsigned int seed_lo, seed_hi; // 64-bit seed split into the two Philox key words

	// Incremented once per add_noise() call so successive frames draw disjoint sub-streams.
	// Atomic because clone() shares the handler between module copies, which the pipeline
	// may run concurrently on separate streams.
	std::atomic<unsigned long long> call_counter;

public:
	explicit Cuda_channel_prng(int dev_id);

	// No allocation and no kernel launch: just re-keys the generator and rewinds the counter.
	void set_seed(unsigned long long seed);

	void add_noise(const float* d_x,
	               float*       d_y,
	               int          n_samples,
	               float        sigma,
	               spu::device_interface::GpuStream* stream);
};

#endif /* Channel_AWGN_LLR_PRNG_CUDA_HPP_ */
