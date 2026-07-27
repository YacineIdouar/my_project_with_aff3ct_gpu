#pragma once

#include <mutex>

namespace sp_vulkan
{

/**
 * Serialises every Vulkan submission made by this example.
 *
 * StreamPU hands the *same* VkQueue to every stream it creates: VULKAN_device::create_stream()
 * passes compute_queue_, obtained once from vkGetDeviceQueue(..., 0, &compute_queue_), to each
 * new VulkanStream. Only the command pool, command buffer and fence are per-stream.
 *
 * VULKAN_executor::dispatch_chain_and_wait() then calls vkQueueSubmit() on that shared queue
 * with no lock (the one mutex in Vulkan_executor.cpp guards the pipeline cache, not the submit).
 * The Vulkan specification lists the queue parameter of vkQueueSubmit as externally
 * synchronised, so two threads submitting at the same time is undefined behaviour.
 *
 * That happens as soon as two Vulkan tasks can run concurrently, which is either:
 *   - a replicated Vulkan stage (set_n_threads(> 1) on the decoder stage), or
 *   - two different Vulkan stages (the channel and the decoder) running at once.
 *
 * Both show up as rare, catastrophically corrupted frames rather than a clean failure.
 *
 * This lock is a workaround at the call sites we own, not a fix: it serialises GPU submissions
 * and therefore gives up the overlap the replication was meant to buy. The proper fix is a
 * mutex around the submit inside StreamPU (or one VkQueue per stream where the device exposes
 * several compute queues).
 */
inline std::mutex& submit_mutex()
{
	static std::mutex m;
	return m;
}

}
