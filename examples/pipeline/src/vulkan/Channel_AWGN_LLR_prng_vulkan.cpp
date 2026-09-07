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

Why the Philox counter is in a buffer and not in the push constants
-------------------------------------------------------------------
StreamPU caches a recorded command buffer under the whole (spirv, buffers, group counts, push
constant *bytes*) tuple, and this dispatch's counter has to change every frame or two frames would
draw the same noise. Putting it in the push constants therefore missed the cache on every single
frame: measured over 200 frames, the channel re-recorded 202 times where the decoder -- whose push
constants are fixed -- recorded once. Each miss is a descriptor pool create, a descriptor set
allocation and a full re-record, and it showed: 138 us of host-side launch for 13 us of GPU work.
Worse, the cache does not evict, so every one of those recordings was also *kept*, growing the
stream's command buffer and descriptor pool count without bound for the length of the run.

So the counter moved into an eight-byte host-visible buffer bound at binding 2 (see
Shaders/awgn_philox.comp). The dispatch key is then fixed for the whole run, the recording is
reused, and the per-frame host work is two stores into a mapped pointer.

This is safe against StreamPU's launch fusion (Device/Gpu_batch.hpp), which can defer a submit: a
batch is opened and closed inside one execution of one sub-sequence on one thread, so anything
referencing the counter buffer has been flushed and waited on before this handler is called again.
It would stop being safe if a batch could ever span two frames.
*/

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

#include "Vulkan/Channel_AWGN_LLR_prng_vulkan.hpp"
#include "Module/Decoder_gpu/gpu_dispatch_mode.hpp"
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

// Layout must match the push_constant block of Shaders/awgn_philox.comp. Everything here is fixed
// for a whole run (sigma changes only between SNR points), which is what keeps the recorded command
// buffer reusable -- the counter that does change per frame lives in a buffer instead.
struct AwgnPushConstants
{
	uint32_t n;
	float    sigma;
	uint32_t seed_lo;
	uint32_t seed_hi;
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

namespace
{
// One described dispatch and the key it was described for. Held by pointer in the cache below,
// never by value in a vector: the descriptor stores the address of the push constants sitting right
// here, so an entry must not move once built.
struct Awgn_entry
{
	// Key: everything the recorded command buffer bakes in. sigma is part of it because it rides in
	// the push constants and moves between SNR points; the two socket buffers are part of it because
	// an adaptor in no-copy mode hands the task a different pair almost every frame.
	VkBuffer in_buf  = VK_NULL_HANDLE;
	VkBuffer out_buf = VK_NULL_HANDLE;
	uint32_t n       = 0;
	float    sigma   = 0.f;
	uint32_t seed_lo = 0;
	uint32_t seed_hi = 0;

	// Referenced by address from the descriptor, so it has to outlive the call that built it.
	AwgnPushConstants pc{};

	std::vector<spu::executor::VULKAN_dispatch_desc> chain;

	bool matches(VkBuffer in, VkBuffer out, uint32_t nn, float sg, uint32_t lo, uint32_t hi) const
	{
		return in_buf == in && out_buf == out && n == nn && sigma == sg && seed_lo == lo &&
		       seed_hi == hi;
	}
};

// Generous, because nothing here is expensive to keep: an entry is a handful of pointers and a
// one-element vector. It exists only so that a caller whose parameters genuinely change every frame
// degrades to re-describing instead of growing without bound. StreamPU's own recording cache, which
// this shadows, is unbounded.
constexpr size_t max_cached_entries = 64;
}

// Named in the header, defined here. One instance per handler, and a handler belongs to one replica
// of the channel module, hence to one thread -- so nothing below is locked.
struct sp_vulkan::Awgn_dispatch_cache
{
	// The counter the shader reads at binding 2: a two-word host-visible allocation, written just
	// before each submit and shared by every entry. 'host' is the mapped pointer, 'buf' the VkBuffer
	// the descriptor sets bind -- and it is 'buf' being *stable* that keeps the recordings valid.
	uint32_t* host = nullptr;
	VkBuffer  buf  = VK_NULL_HANDLE;

	std::vector<std::unique_ptr<Awgn_entry>> entries;
	size_t last = 0; // entry the previous call used, tried first

	// Used for a key arriving once the cache is full, and re-described every time. Never evicting a
	// cached entry is deliberate: the sockets rotate cyclically, and on a cycle longer than the
	// cache any policy that recycles entries discards exactly the one needed next.
	std::unique_ptr<Awgn_entry> overflow;

	uint64_t builds = 0; // diagnostics only (SPU_DEBUG_VULKAN)
	uint64_t hits   = 0;
};

sp_vulkan::Vulkan_channel_prng::Vulkan_channel_prng(int device_id)
  : dev_id(device_id), seed_lo(42u), seed_hi(0u), call_counter(0ull)
{
}

sp_vulkan::Vulkan_channel_prng::~Vulkan_channel_prng()
{
	// Only the host-side description is ours to free: the counter allocation belongs to
	// Devices_manager and goes with the device, and the recorded command buffers belong to the
	// VulkanStream.
	delete this->cache;
	this->cache = nullptr;
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

	if (this->cache == nullptr) this->cache = new Awgn_dispatch_cache;
	Awgn_dispatch_cache& c = *this->cache;

	// Two words of host-visible memory, allocated once. memory_type::HOST is what makes the returned
	// pointer dereferenceable (see VULKAN_device::allocate_memory); a DEVICE allocation would need a
	// staged copy per frame, which is exactly the round trip this is here to avoid.
	if (c.host == nullptr)
	{
		uint8_t* p = spu::Devices_manager::allocate_memory(
			2 * sizeof(uint32_t),
			{ spu::device_interface::memory_type::HOST, spu::device_interface::compute_api::VULKAN,
			  device_id, 0 });
		c.host = reinterpret_cast<uint32_t*>(p);
		c.buf  = spu::Devices_manager::get_vulkan_buffer(device_id, p);
	}

	VkBuffer in_buf  = spu::Devices_manager::get_vulkan_buffer(
		device_id, reinterpret_cast<uint8_t*>(const_cast<float*>(d_x)));
	VkBuffer out_buf = spu::Devices_manager::get_vulkan_buffer(
		device_id, reinterpret_cast<uint8_t*>(d_y));

	const uint32_t n  = (uint32_t)n_samples;
	const uint32_t lo = this->seed_lo;
	const uint32_t hi = this->seed_hi;

	// Find the description for this key, or build it. The fast path is the entry the previous call
	// used; the scan behind it catches the socket buffers rotating over an adaptor's pool, which is
	// the common case here and would otherwise re-describe on every frame.
	Awgn_entry* e = nullptr;
	if (c.last < c.entries.size() && c.entries[c.last]->matches(in_buf, out_buf, n, sigma, lo, hi))
	{
		e = c.entries[c.last].get();
		c.hits++;
	}
	else
	{
		for (size_t i = 0; i < c.entries.size() && e == nullptr; ++i)
			if (c.entries[i]->matches(in_buf, out_buf, n, sigma, lo, hi))
			{
				c.last = i;
				e = c.entries[i].get();
				c.hits++;
			}
	}

	if (e == nullptr)
	{
		if (c.entries.size() < max_cached_entries)
		{
			c.entries.push_back(std::unique_ptr<Awgn_entry>(new Awgn_entry));
			c.last = c.entries.size() - 1;
			e = c.entries[c.last].get();
		}
		else
		{
			if (c.overflow == nullptr) c.overflow.reset(new Awgn_entry);
			c.last = c.entries.size(); // keep the fast path above from claiming a stale hit
			e = c.overflow.get();
		}

		const uint32_t active_threads = (n + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD;
		const uint32_t groups         = (active_threads + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

		const auto& spirv = get_awgn_philox_spirv_words();

		e->pc = AwgnPushConstants{ n, sigma, lo, hi };

		e->chain.clear();
		e->chain.push_back({ spirv.data(), spirv.size() * sizeof(uint32_t),
		                     { in_buf, out_buf, c.buf }, groups, 1u,
		                     &e->pc, sizeof(e->pc) });

		e->in_buf  = in_buf;
		e->out_buf = out_buf;
		e->n       = n;
		e->sigma   = sigma;
		e->seed_lo = lo;
		e->seed_hi = hi;
		c.builds++;

		if (std::getenv("SPU_DEBUG_VULKAN"))
			std::fprintf(stderr,
			             "[awgn] described the dispatch (%llu described, %llu reused, %zu entries)\n",
			             (unsigned long long)c.builds,
			             (unsigned long long)c.hits,
			             c.entries.size());
	}

	/* One counter value per dispatch, claimed atomically: this is the whole "state" of the
	   generator, and it lives on the host, so no device memory is touched between frames. */
	const unsigned long long counter = this->call_counter.fetch_add(1ull);

	// The only per-frame write. Safe to do unsynchronised: the previous dispatch of this handler
	// fence-waited before add_noise() returned, so nothing on the GPU is still reading these words.
	c.host[0] = (uint32_t)(counter & 0xFFFFFFFFull);
	c.host[1] = (uint32_t)(counter >> 32);
	// No-op on the coherent allocations every platform here gives; it matters only if HOST landed on
	// a cached, non-coherent memory type.
	spu::Devices_manager::flush_vulkan_memory(device_id, reinterpret_cast<uint8_t*>(c.host));

	spu::executor::VULKAN_executor exec(vulkan_stream->device());
	exec.set_stream(vulkan_stream);

	// --gpu-dispatch is applied through SPU_VULKAN_DISPATCH_MODE (see gpu_dispatch_mode.hpp):
	// StreamPU has no per-executor setter any more, the mode being a property of the whole run.

	// --dec-profile / SPU_LDPC_PROFILE. Forcing the executor's own flag to match ours keeps the two
	// from disagreeing: get_timings() reports what was actually recorded. No lock needed
	// around the executor here -- unlike the CUDA/HIP channels, this one is built per call.
	const bool profiled = gpu_prof::enabled();
	exec.set_profiling(profiled);

	// The flush of d_x and the invalidate of d_y that used to bracket this dispatch have been
	// removed. They only did anything when allocate_memory() fell back to a cached, non-coherent
	// memory type (see VULKAN_device::allocate_memory / flush_memory / invalidate_memory -- both
	// are no-ops on coherent allocations, which is what a unified-memory part like Orin gives).
	// Restore them if a platform ever reports corrupted samples.
	//
	// No submit lock either: StreamPU takes the stream's queue mutex itself, around vkQueueSubmit
	// alone, and deliberately leaves the fence wait outside it (see VULKAN_executor::
	// submit_and_wait). Wrapping the whole launch as this used to would put the GPU wait inside a
	// mutex shared with the decoder, serialising the two stages on execution rather than on the
	// submit call.
	if (profiled) exec.launch_profiled(e->chain);
	else          exec.launch(e->chain);

	// One record for the one dispatch: (host-side launch, GPU time of the shader).
	if (profiled) this->prof.add(exec.get_timings());
}
