#ifndef Channel_AWGN_LLR_CUDA_HPP_
#define Channel_AWGN_LLR_CUDA_HPP_

#include <streampu.hpp>
#include "Module/Decoder_gpu/gpu_decoder_profiling.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

namespace spu { namespace executor { class CUDA_executor; } }

class Cuda_channel
{
private:
	int dev_id;
	spu::executor::CUDA_executor* executor;
	// Per-handler profiling totals, registered process-wide; see
	// Module/Decoder_gpu/gpu_decoder_profiling.hpp.
	gpu_prof::accumulator prof{ "CHANNEL", "CUDA" };


	// Deliberately void* and deliberately *not* behind #ifdef __CUDACC__.
	//
	// It used to be a curandStateXORWOW_t* declared only when __CUDACC__ was defined, which made
	// sizeof(Cuda_channel) differ between this file's two readers: nvcc compiles the constructor
	// with the member, the host compiler builds Channel_AWGN_LLR_gpu.cpp -- which does
	// `new Cuda_channel(dev_id)` -- without it. The constructor then wrote past the end of an
	// allocation one pointer too small. It happened to land in allocator slack and stayed invisible
	// until another member was added ahead of it, at which point the run ended in
	// 'corrupted size vs. prev_size'. One unconditional pointer, cast where it is used, keeps every
	// translation unit agreeing on the layout.
	void* d_states;
public:
	Cuda_channel(int dev_id);
	void init_rand_state(int max_samples, int seed = 42, int threads = 256);
	void add_noise(const float*  d_x,
               float*        d_y,
               int           n_samples,
               float         sigma,
			   spu::device_interface::GpuStream* stream);
};

#endif