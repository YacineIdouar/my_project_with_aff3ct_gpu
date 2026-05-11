#ifndef Channel_AWGN_LLR_CUDA_HPP_
#define Channel_AWGN_LLR_CUDA_HPP_

#include "Device/Cuda/Cuda_device.hpp"

void init_rand_state(int max_samples, int seed = 42, int threads = 256);
void add_noise(const float*  d_x,
               float*        d_y,
               int           n_samples,
               float         sigma,
			   spu::sp_cuda::CudaStream spu_stream);

#endif