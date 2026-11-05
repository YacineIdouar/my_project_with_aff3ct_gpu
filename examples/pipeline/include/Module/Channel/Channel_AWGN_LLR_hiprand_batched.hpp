#ifndef Channel_AWGN_LLR_HIPRAND_BATCHED_HPP_
#define Channel_AWGN_LLR_HIPRAND_BATCHED_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Hip/Channel_AWGN_LLR_hiprand_batched_hip.hpp"

namespace aff3ct
{
namespace module
{
/**
 * hipRAND AWGN channel on HIP written with TidePU's Generic module: the HIP counterpart of
 * Channel_AWGN_LLR_curand_batched, with hipRAND's XORWOW generator in place of cuRAND's.
 *
 * Same task and sockets as the other channels of this example ('add_noise_gpu': 'CP', 'X_N', 'Y_N'). The noise is
 * element-wise and the frames do not depend on each other, so the implementation is registered with
 * 'gpu_batched()': it receives the n_f frames of a firing stored back to back and adds noise to all of them in one
 * launch (one hipRAND state per pair of samples of the firing).
 *
 * The hipRAND states are device state of each replica ('per_device'): created by on_device(), re-sized by
 * on_n_frames(), re-seeded by set_seed().
 *
 * A replica builds them when it is cloned (on_clone()), not lazily at its first firing as StreamPU would: seeding
 * ends with a hipDeviceSynchronize(), after which the HIP runtime invalidates every stream capture in flight in the
 * process, whatever its capture mode -- so a replica setting up would break the graph another GPU task is capturing
 * ("operation failed due to a previous error during capture"). Built ahead of the run, nothing of it runs while the
 * pipeline does.
 */
class Channel_AWGN_LLR_hiprand_batched final : public spu::module::Generic<Channel_AWGN_LLR_hiprand_batched>
{
  private:
    int N; // samples per frame
    uint64_t seed;
    spu::module::per_device<sp_hip::Hip_channel_batched> dev;

  public:
    Channel_AWGN_LLR_hiprand_batched(const int N, const uint64_t seed = 42);

    int get_N() const;

    void on_device(const api exec_api, const int device_id, const int platform_id) override;
    void on_n_frames(const size_t n_frames) override;
    void on_clone(const size_t replica_id) override;
    void set_seed(const int seed) override;

    // The n_f frames of the firing in one launch. 'CP' holds one sigma per frame; the frames of a firing come from
    // the same SNR point, so one value is passed to the kernel (checked).
    std::vector<launch_timing> add_noise_hip(span<const float> CP, span<const float> X_N, span<float> Y_N, call c);
};
}
}

#endif /* Channel_AWGN_LLR_HIPRAND_BATCHED_HPP_ */
