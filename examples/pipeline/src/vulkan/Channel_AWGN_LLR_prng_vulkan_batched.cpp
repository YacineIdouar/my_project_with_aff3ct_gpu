/*
Channel_AWGN_LLR_prng_vulkan_batched.cpp
----------------------------------------
Vulkan part of Channel_AWGN_LLR_prng_vulkan_batched: the Philox AWGN pass of Channel_AWGN_LLR_prng_vulkan.cpp over all
the samples of a firing, then the device-side counter step, as one chain (one submit, one wait).
*/

#ifdef DECODER_VULKAN

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include "Vulkan/Channel_AWGN_LLR_prng_vulkan_batched.hpp"
#include "Device/Devices_manager.hpp"
#include "Device/Vulkan/Vulkan_device.hpp"
#include "Device/Vulkan/Vulkan_dispatch.hpp"
#include "Device/Vulkan/Vulkan_executor.hpp"

#include "awgn_philox_spv.hpp"         // generated at build time from Shaders/awgn_philox.comp by glslc
#include "philox_counter_step_spv.hpp" // generated at build time from Shaders/philox_counter_step.comp by glslc

using spu::device_interface::compute_api;
using spu::device_interface::memory_type;

namespace
{
// Must match local_size_x and SAMPLES_PER_THREAD in Shaders/awgn_philox.comp, and Channel_AWGN_LLR_prng_vulkan.cpp:
// the noise a sample gets depends on the dispatch shape.
constexpr uint32_t THREADS_PER_GROUP = 256;
constexpr uint32_t SAMPLES_PER_THREAD = 4;

// Layouts of the push_constant blocks of Shaders/awgn_philox.comp and Shaders/philox_counter_step.comp.
struct AwgnPushConstants
{
    uint32_t n;
    float sigma;
    uint32_t seed_lo;
    uint32_t seed_hi;
};
struct StepPushConstants
{
    uint32_t step;
};

VkBuffer
buffer_of(const int dev_id, const void* ptr)
{
    return spu::Devices_manager::get_vulkan_buffer(dev_id, static_cast<uint8_t*>(const_cast<void*>(ptr)));
}
}

sp_vulkan::Vulkan_channel_batched::Vulkan_channel_batched(const int dev_id, const uint64_t seed)
  : dev_id(dev_id)
  , seed(seed)
{
    this->counter = reinterpret_cast<uint32_t*>(spu::Devices_manager::allocate_memory(
      2 * sizeof(uint32_t), { memory_type::DEVICE, compute_api::VULKAN, this->dev_id, 0 }));
    this->set_seed(seed);
}

sp_vulkan::Vulkan_channel_batched::~Vulkan_channel_batched()
{
    spu::Devices_manager::deallocate_memory(reinterpret_cast<uint8_t*>(this->counter),
                                            { memory_type::DEVICE, compute_api::VULKAN, this->dev_id, 0 });
}

void
sp_vulkan::Vulkan_channel_batched::set_seed(const uint64_t seed)
{
    this->seed = seed;
    // setup, not part of a firing: nothing on the device is reading the counter
    const uint32_t zero[2] = { 0u, 0u };
    spu::Devices_manager::upload_vulkan_memory(
      this->dev_id, reinterpret_cast<uint8_t*>(this->counter), zero, sizeof(zero));
}

std::vector<spu::device_interface::launch_timing>
sp_vulkan::Vulkan_channel_batched::add_noise(const float* x,
                                             float* y,
                                             const size_t n,
                                             const float sigma,
                                             spu::device_interface::GpuStream* stream)
{
    // static: the address of the words is part of the executor's pipeline cache key
    static const auto awgn_words =
      spu::sp_vulkan::spirv_words(pipeline_vulkan_shaders::awgn_philox_spv, pipeline_vulkan_shaders::awgn_philox_spv_size);
    static const auto step_words = spu::sp_vulkan::spirv_words(pipeline_vulkan_shaders::philox_counter_step_spv,
                                                               pipeline_vulkan_shaders::philox_counter_step_spv_size);

    const uint32_t active_threads = (uint32_t)((n + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD);
    const uint32_t groups = (active_threads + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

    // fixed for a whole SNR point: the chain's recording is reused from one firing to the next
    const AwgnPushConstants awgn_pc{
        (uint32_t)n, sigma, (uint32_t)(this->seed & 0xFFFFFFFFull), (uint32_t)(this->seed >> 32)
    };
    const StepPushConstants step_pc{ 1u };

    const VkBuffer counter_buf = buffer_of(this->dev_id, this->counter);
    const std::vector<spu::executor::VULKAN_dispatch_desc> chain = {
        { awgn_words.data(),
          awgn_words.size() * sizeof(uint32_t),
          { buffer_of(this->dev_id, x), buffer_of(this->dev_id, y), counter_buf },
          groups,
          1u,
          &awgn_pc,
          sizeof(awgn_pc) },
        { step_words.data(), step_words.size() * sizeof(uint32_t), { counter_buf }, 1u, 1u, &step_pc, sizeof(step_pc) }
    };

    auto* vulkan_stream = static_cast<spu::sp_vulkan::VulkanStream*>(stream);
    spu::executor::VULKAN_executor exec(vulkan_stream->device());
    exec.set_stream(vulkan_stream);
    exec.launch_profiled(chain);
    return exec.get_timings();
}

#endif // DECODER_VULKAN
