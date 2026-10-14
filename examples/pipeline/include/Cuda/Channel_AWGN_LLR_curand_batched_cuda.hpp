#ifndef Channel_AWGN_LLR_CURAND_BATCHED_CUDA_HPP_
#define Channel_AWGN_LLR_CURAND_BATCHED_CUDA_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

/**
 * cuRAND XORWOW states on one device, one per pair of samples, and the AWGN kernel that consumes them (the
 * Cuda_channel of Channel_AWGN_LLR_cuda.hpp, launched through StreamPU's sp_cuda::dispatch()).
 *
 * The states are a void*, as in Cuda_channel, so that nvcc and the host compiler agree on the class layout.
 */
class Cuda_channel_batched
{
  private:
    int dev_id;
    void* d_states = nullptr; // curandStateXORWOW_t[n_states]
    size_t n_states = 0;

  public:
    // Allocates and seeds the states for up to 'max_samples' samples per call.
    Cuda_channel_batched(int dev_id, size_t max_samples, uint64_t seed);
    ~Cuda_channel_batched();
    Cuda_channel_batched(const Cuda_channel_batched&) = delete;
    Cuda_channel_batched& operator=(const Cuda_channel_batched&) = delete;

    // (Re-)allocates the states for 'max_samples' samples and seeds them: each state starts its own sequence.
    void init(size_t max_samples, uint64_t seed);

    // Samples a call can process.
    size_t capacity() const { return 2 * this->n_states; }

    // y[i] = x[i] + sigma * N(0,1) for i < n, through the executor of the task's stream.
    std::vector<spu::device_interface::launch_timing> add_noise(const float* x,
                                                                float* y,
                                                                size_t n,
                                                                float sigma,
                                                                spu::device_interface::GpuStream* stream);
};

#endif /* Channel_AWGN_LLR_CURAND_BATCHED_CUDA_HPP_ */
