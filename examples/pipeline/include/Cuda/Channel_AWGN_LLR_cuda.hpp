#ifndef Channel_AWGN_LLR_CUDA_HPP_
#define Channel_AWGN_LLR_CUDA_HPP_

#include <streampu.hpp>

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
#ifdef __CUDACC__
	curandStateXORWOW_t* d_states;
#endif
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