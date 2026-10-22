#ifndef Channel_AWGN_LLR_PRNG_VULKAN_BATCHED_HPP_
#define Channel_AWGN_LLR_PRNG_VULKAN_BATCHED_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Vulkan/Channel_AWGN_LLR_prng_vulkan_batched.hpp"

namespace aff3ct
{
namespace module
{
/**
 * Philox AWGN channel on Vulkan (port of Channel_AWGN_LLR_prng_gpu's Vulkan backend) written with TidePU's Generic
 * module, the Vulkan counterpart of Channel_AWGN_LLR_curand_batched.
 *
 * Same task and sockets as the other channels of this example ('add_noise_gpu': 'CP', 'X_N', 'Y_N'). The noise is
 * element-wise and the frames do not depend on each other, so the implementation is registered with
 * 'gpu_batched()': it receives the n_f frames of a firing stored back to back and adds noise to all of them in one
 * dispatch.
 *
 * The Philox counter is device state of each replica ('per_device'), rewound by set_seed(). The generator is
 * counter-based, so unlike cuRAND's states nothing depends on the number of frames.
 *
 * Unlike the CUDA modules, the device state is built with the module -- by the constructor, and by on_clone() for a
 * replica, for the device given to the constructor -- rather than by on_device(). on_device() runs on the thread of
 * the module's stage, at its first firing, so the replicas and the other stages would allocate concurrently, and
 * StreamPU's Vulkan allocator is not thread-safe (VULKAN_device's allocation map has no lock): the lookups of every
 * stage then read a corrupted map. The constructor and on_clone() both run on the thread that builds the pipeline.
 * on_device() keeps that state when the task runs on that device, and only builds another one for another device.
 */
class Channel_AWGN_LLR_prng_vulkan_batched final : public spu::module::Generic<Channel_AWGN_LLR_prng_vulkan_batched>
{
  private:
    int N; // samples per frame
    uint64_t seed;
    int device_id; // the Vulkan device 'dev' is built for
    spu::module::per_device<sp_vulkan::Vulkan_channel_batched> dev;

  public:
    Channel_AWGN_LLR_prng_vulkan_batched(const int N, const uint64_t seed = 42, const int device_id = 0);

    int get_N() const;

    void on_clone(const size_t replica_id) override;
    void on_device(const api exec_api, const int device_id, const int platform_id) override;
    void set_seed(const int seed) override;

    // The n_f frames of the firing in one dispatch. 'CP' holds one sigma per frame; the frames of a firing come from
    // the same SNR point, so one value is passed to the shader (checked).
    std::vector<launch_timing> add_noise_vulkan(span<const float> CP, span<const float> X_N, span<float> Y_N, call c);
};
}
}

#endif /* Channel_AWGN_LLR_PRNG_VULKAN_BATCHED_HPP_ */
