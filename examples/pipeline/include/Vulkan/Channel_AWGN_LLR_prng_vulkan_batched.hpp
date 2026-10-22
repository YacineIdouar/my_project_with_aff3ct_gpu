#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

namespace sp_vulkan
{
/**
 * Philox4x32-10 AWGN channel on one Vulkan device (the generator of Vulkan_channel_prng, Shaders/awgn_philox.comp),
 * dispatched once over all the samples of a firing.
 *
 * The generator is counter-based, so the only state is its 64-bit counter, and it lives on the device: awgn_philox.comp
 * reads it and philox_counter_step.comp, the second dispatch of the chain, advances it. Vulkan_channel_prng writes it
 * from the host before each submit instead, which is only safe while every launch completes before the next one is
 * described; on the device it stays right when a firing is launched before the previous one has run (an asynchronous
 * stage), and the chain's key stays the same for the whole run, so its recorded command buffer is reused.
 *
 * For a firing of one frame, the samples, the counter and the dispatch geometry are those of Vulkan_channel_prng: the
 * same noise, call after call.
 *
 * Holds no Vulkan type, so that the module's translation unit (obj-module) does not need vulkan.h.
 */
class Vulkan_channel_batched
{
  private:
    int dev_id;
    uint64_t seed;
    uint32_t* counter = nullptr; // two words of device memory: low, high

  public:
    Vulkan_channel_batched(int dev_id, uint64_t seed);
    ~Vulkan_channel_batched();
    Vulkan_channel_batched(const Vulkan_channel_batched&) = delete;
    Vulkan_channel_batched& operator=(const Vulkan_channel_batched&) = delete;

    // Re-keys the generator and rewinds the counter: the next call draws the first sub-stream of 'seed'.
    void set_seed(uint64_t seed);

    // y[i] = x[i] + sigma * N(0,1) for i < n, through the executor of the task's stream.
    std::vector<spu::device_interface::launch_timing> add_noise(const float* x,
                                                                float* y,
                                                                size_t n,
                                                                float sigma,
                                                                spu::device_interface::GpuStream* stream);
};
}
