#ifndef Channel_AWGN_LLR_CURAND_BATCHED_HPP_
#define Channel_AWGN_LLR_CURAND_BATCHED_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Cuda/Channel_AWGN_LLR_curand_batched_cuda.hpp"

namespace aff3ct
{
namespace module
{
/**
 * cuRAND AWGN channel on CUDA (port of Channel_AWGN_LLR_gpu) written with TidePU's Generic module.
 *
 * Same task and sockets as the other channels of this example ('add_noise_gpu': 'CP', 'X_N', 'Y_N'). The noise is
 * element-wise and the frames do not depend on each other, so the implementation is registered with
 * 'gpu_batched()': it receives the n_f frames of a firing stored back to back and adds noise to all of them in one
 * launch (one cuRAND state per pair of samples of the firing, as in the original module).
 *
 * The cuRAND states are device state of each replica ('per_device'): created by on_device(), re-sized by
 * on_n_frames(), re-seeded by set_seed().
 */
class Channel_AWGN_LLR_curand_batched final : public spu::module::Generic<Channel_AWGN_LLR_curand_batched>
{
  private:
    int N; // samples per frame
    uint64_t seed;
    spu::module::per_device<Cuda_channel_batched> dev;

  public:
    Channel_AWGN_LLR_curand_batched(const int N, const uint64_t seed = 42);

    int get_N() const;

    void on_device(const api exec_api, const int device_id, const int platform_id) override;
    void on_n_frames(const size_t n_frames) override;
    void set_seed(const int seed) override;

    // The n_f frames of the firing in one launch. 'CP' holds one sigma per frame; the frames of a firing come from
    // the same SNR point, so one value is passed to the kernel (checked).
    std::vector<launch_timing> add_noise_cuda(span<const float> CP, span<const float> X_N, span<float> Y_N, call c);
};
}
}

#endif /* Channel_AWGN_LLR_CURAND_BATCHED_HPP_ */
