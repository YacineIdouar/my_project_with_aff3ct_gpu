/**
 * Channel_AWGN_LLR_curand_batched_cuda.cu
 * ---------------------------------------
 * cuRAND AWGN channel of Channel_AWGN_LLR_curand_batched (port of Channel_AWGN_LLR_cuda.cu): one thread per pair of
 * samples, curand_normal2() for the Box-Muller pair, the state written back so that the next call continues its
 * sequence. The module hands it all the frames of a firing at once, as one array.
 */

#include <cstdint>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Cuda/Channel_AWGN_LLR_curand_batched_cuda.hpp"
#include "Device/Cuda/Cuda_dispatch.hpp"

namespace
{
constexpr unsigned THREADS = 256;

void
check(const cudaError_t error, const char* what)
{
    if (error == cudaSuccess) return;
    std::stringstream message;
    message << what << ": " << cudaGetErrorString(error) << ".";
    throw spu::tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
}

// one state per pair of samples; 'sequence = idx' makes the states statistically independent
__global__ void
init_states_kernel(curandStateXORWOW_t* __restrict__ states, const uint32_t n_states, const unsigned long long seed)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) curand_init(seed, (unsigned long long)idx, 0ull, &states[idx]);
}

// y = x + sigma * n, n ~ N(0,1), two samples per thread
__global__ void
awgn_add_noise_kernel(const float* __restrict__ x,
                      float* __restrict__ y,
                      const uint32_t n,
                      const float sigma,
                      curandStateXORWOW_t* __restrict__ states)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t base = idx * 2;
    if (base >= n) return;

    curandStateXORWOW_t local_state = states[idx];
    const float2 noise = curand_normal2(&local_state);
    y[base] = x[base] + sigma * noise.x;
    if (base + 1 < n) y[base + 1] = x[base + 1] + sigma * noise.y;
    states[idx] = local_state;
}
}

Cuda_channel_batched::Cuda_channel_batched(const int dev_id, const size_t max_samples, const uint64_t seed)
  : dev_id(dev_id)
{
    this->init(max_samples, seed);
}

Cuda_channel_batched::~Cuda_channel_batched()
{
    if (this->d_states != nullptr)
    {
        cudaSetDevice(this->dev_id);
        cudaFree(this->d_states);
    }
}

void
Cuda_channel_batched::init(const size_t max_samples, const uint64_t seed)
{
    check(cudaSetDevice(this->dev_id), "cudaSetDevice");
    const size_t n_states = (max_samples + 1) / 2;
    if (n_states != this->n_states)
    {
        if (this->d_states != nullptr) check(cudaFree(this->d_states), "cudaFree (cuRAND states)");
        this->d_states = nullptr;
        check(cudaMalloc(&this->d_states, n_states * sizeof(curandStateXORWOW_t)), "cudaMalloc (cuRAND states)");
        this->n_states = n_states;
    }

    // setup, not part of a firing: launched directly and waited for
    const unsigned blocks = (unsigned)((n_states + THREADS - 1) / THREADS);
    init_states_kernel<<<blocks, THREADS>>>(
      static_cast<curandStateXORWOW_t*>(this->d_states), (uint32_t)n_states, (unsigned long long)seed);
    check(cudaGetLastError(), "init_states_kernel launch");
    check(cudaDeviceSynchronize(), "init_states_kernel");
}

std::vector<spu::device_interface::launch_timing>
Cuda_channel_batched::add_noise(const float* x,
                                float* y,
                                const size_t n,
                                const float sigma,
                                spu::device_interface::GpuStream* stream)
{
    const uint32_t pairs = (uint32_t)((n + 1) / 2);
    const unsigned blocks = (pairs + THREADS - 1) / THREADS;
    return spu::sp_cuda::dispatch(stream,
                                  awgn_add_noise_kernel,
                                  blocks,
                                  THREADS,
                                  0,
                                  x,
                                  y,
                                  (uint32_t)n,
                                  sigma,
                                  static_cast<curandStateXORWOW_t*>(this->d_states));
}
