/*
Vulkan port of Cuda_channel_prng (see src/cuda/Channel_AWGN_LLR_prng_cuda.cu).

The noise generator is the hand-written Philox4x32-10 of include/Rng/Philox4x32.hpp, ported to
GLSL in Shaders/philox4x32.glsl and dispatched by Shaders/awgn_philox.comp. No library is
involved on either side, which is the whole point: the same generator can be expressed in every
API the project targets.

Unlike the decoder, one AWGN pass is a single dispatch, so there is no chain to build -- but the
submit path is the same VULKAN_executor one, and the flush/invalidate pair around it is needed
for the same reason: X_N is written by the upstream native task (the modem's modulate) and Y_N is
read by the downstream one (demodulate), so the GPU must see the input and the CPU must see the
output even when the allocation landed on non-coherent memory.
*/

#include <cstdint>
#include <cstring>
#include <vector>
#include <vulkan/vulkan.h>

#include "Vulkan/Channel_AWGN_LLR_prng_vulkan.hpp"
#include "Vulkan/vulkan_submit_lock.hpp"
#include "Device/Devices_manager.hpp"
#include "Device/Vulkan/Vulkan_device.hpp"
#include "Device/Vulkan/Vulkan_executor.hpp"

#include "awgn_philox_spv.hpp" // generated at build time from Shaders/awgn_philox.comp by glslc

namespace
{
// Must match local_size_x and SAMPLES_PER_THREAD in Shaders/awgn_philox.comp, and the CUDA
// kernel's launch geometry: the noise a sample gets depends on the dispatch shape, so the two
// backends only agree while these do.
const uint32_t THREADS_PER_GROUP   = 256;
const uint32_t SAMPLES_PER_THREAD  = 4;

// Layout must match the push_constant block of Shaders/awgn_philox.comp.
struct AwgnPushConstants
{
	uint32_t n;
	float    sigma;
	uint32_t seed_lo;
	uint32_t seed_hi;
	uint32_t ctr_lo;
	uint32_t ctr_hi;
};

// The embedded SPIR-V blob is stored as raw bytes (see the generated *_spv.hpp header); Vulkan
// requires the code as a uint32_t* though, so it is copied once, on first use, into a properly
// typed/aligned buffer (a reinterpret_cast from uint8_t* to uint32_t* would be undefined
// behaviour). Mirrors get_clip_spirv_words() in Decoder_LDPC_vulkan_kernel.cpp.
const std::vector<uint32_t>& get_awgn_philox_spirv_words()
{
	static const std::vector<uint32_t> words = []
	{
		std::vector<uint32_t> w(pipeline_vulkan_shaders::awgn_philox_spv_size / sizeof(uint32_t));
		std::memcpy(w.data(), pipeline_vulkan_shaders::awgn_philox_spv,
		            pipeline_vulkan_shaders::awgn_philox_spv_size);
		return w;
	}();
	return words;
}
}

sp_vulkan::Vulkan_channel_prng::Vulkan_channel_prng(int device_id)
  : dev_id(device_id), seed_lo(42u), seed_hi(0u), call_counter(0ull)
{
}

void
sp_vulkan::Vulkan_channel_prng::set_seed(unsigned long long seed)
{
	this->seed_lo = (uint32_t)(seed & 0xFFFFFFFFull);
	this->seed_hi = (uint32_t)(seed >> 32);
	this->call_counter.store(0ull);
}

/**
 * Add AWGN to d_x and write result to d_y.
 *
 * @param d_x        Device input vector
 * @param d_y        Device output vector
 * @param n_samples  Number of float samples
 * @param sigma      Noise standard deviation
 * @param stream     StreamPU GPU stream the dispatch is submitted on
 */
void
sp_vulkan::Vulkan_channel_prng::add_noise(const float* d_x,
                                          float*       d_y,
                                          int          n_samples,
                                          float        sigma,
                                          spu::device_interface::GpuStream* stream)
{
	if (n_samples <= 0) return;

	auto vulkan_stream = static_cast<spu::sp_vulkan::VulkanStream*>(stream);
	const int device_id = this->dev_id;

	VkBuffer in_buf  = spu::Devices_manager::get_vulkan_buffer(
		device_id, reinterpret_cast<uint8_t*>(const_cast<float*>(d_x)));
	VkBuffer out_buf = spu::Devices_manager::get_vulkan_buffer(
		device_id, reinterpret_cast<uint8_t*>(d_y));

	const uint32_t active_threads = ((uint32_t)n_samples + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD;
	const uint32_t groups         = (active_threads + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

	/* One counter value per dispatch, claimed atomically: this is the whole "state" of the
	   generator, and it lives on the host, so no device memory is touched between frames. */
	const unsigned long long counter = this->call_counter.fetch_add(1ull);

	AwgnPushConstants pc{ (uint32_t)n_samples,
	                      sigma,
	                      this->seed_lo,
	                      this->seed_hi,
	                      (uint32_t)(counter & 0xFFFFFFFFull),
	                      (uint32_t)(counter >> 32) };

	const auto& spirv = get_awgn_philox_spirv_words();

	std::vector<spu::executor::VULKAN_dispatch_desc> chain;
	chain.push_back({ spirv.data(), spirv.size() * sizeof(uint32_t),
	                  { in_buf, out_buf }, groups, 1u,
	                  &pc, sizeof(pc) });

	spu::executor::VULKAN_executor exec(vulkan_stream->device());
	exec.set_stream(vulkan_stream);

	spu::Devices_manager::flush_vulkan_memory(
		device_id, reinterpret_cast<uint8_t*>(const_cast<float*>(d_x)));

	{
		// Every VulkanStream shares one VkQueue; vkQueueSubmit() needs external synchronisation.
		std::lock_guard<std::mutex> submit_lock(sp_vulkan::submit_mutex());
		exec.dispatch_chain_and_wait(chain);
	}

	spu::Devices_manager::invalidate_vulkan_memory(device_id, reinterpret_cast<uint8_t*>(d_y));
}
