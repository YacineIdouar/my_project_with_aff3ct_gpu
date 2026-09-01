/*
Vulkan port of the CUDA/HIP/SYCL LDPC 5G decoder -- variant built around StreamPU's recorded
dispatch cache (see Decoder_LDPC_vulkan_kernel.cpp for the rebuild-every-call version this
replaces, and VulkanStream::find_recorded_dispatch() in
lib/aff3ct/lib/streampu/include/Device/Vulkan/Vulkan_device.hpp for the cache itself).

Both implementations are always compiled and both derive from sp_vulkan::Vulkan_decoder; which one
a run uses is decided at startup by Vulkan_decoder::create() (see Decoder_LDPC_vulkan_decoder.cpp),
driven by --dec-vk-chain in main_gpu.cpp or by SPU_LDPC_VULKAN_CHAIN. This one is the default.

What changes vs the original
----------------------------
StreamPU records the whole dispatch chain into a VkCommandBuffer and keeps it on the stream, keyed
on the *sequence* of (spirv blob address, buffers, group counts, push constant bytes). When the
same key comes back, a frame costs one vkQueueSubmit plus the fence wait -- no descriptor pool, no
descriptor sets, no re-recording. That lookup is automatic: the original file already hits it,
because everything it passes happens to be identical from one decode to the next.

What the original still pays on every single frame, though, is the *host-side* work of describing
the chain before the lookup can even happen: a std::vector<VULKAN_dispatch_desc> of 2 + 2*n_ite
entries, each carrying its own heap-allocated std::vector<VkBuffer>. At n_ite = 20 that is 42
descriptors and 43 allocations rebuilt per decoded frame, thrown away right after, only for
dispatch_impl() to memcmp them against a chain that has not moved since the first frame.

So this variant caches the description too. The chain is built once per decoder instance and kept
alive in the thread context; a decode call validates it against the parameters that could have
changed (the two socket buffers, num_iter, K) and, on the overwhelmingly common hit, goes straight
to dispatch_chain_and_wait() with the vector it already owns. The per-frame host path is then just
the key-view build inside dispatch_chain_and_wait() plus the stream's cache scan.

Keeping the chain alive is what forces the rest of the design: VULKAN_dispatch_desc stores its
push constants as a bare `const void*`, so the structs behind those pointers have to outlive the
call that recorded them -- they live in the context here, not on ldpc_decode()'s stack.

The cache holds a small set of chains rather than just the last one, for the same reason StreamPU's
own cache does. Adaptor_m_to_n in no-copy mode -- the default -- swaps a task's socket buffer with
one from its pool every frame, so `llr_in`/`llr_bits_out` rotate over a handful of allocations and
a one-entry cache would thrash on every frame while StreamPU's eight-entry one still hit. Passing
`--dec-stage-copy` to pipeline_gpu takes the decoder stage out of no-copy mode, which pins both
sockets for the whole run and collapses this to a single entry described once.

Set SPU_DEBUG_VULKAN to watch it: each describe prints the running (described, reused, entries)
counts, and StreamPU prints its own recordings next to them. In copy mode both settle at one per
decoder thread and never move again.
*/

#ifdef DECODER_VULKAN

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

#include "Vulkan/Decoder_LDPC_vulkan_kernel.hpp"
#include "Vulkan/vulkan_submit_lock.hpp"
#include "Module/Decoder_gpu/gpu_dispatch_mode.hpp"
#include "Device/Devices_manager.hpp"
#include "Device/Vulkan/Vulkan_device.hpp"
#include "Device/Vulkan/Vulkan_executor.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg1.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg2.hpp"
#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"

#include "ldpc_clip_channel_spv.hpp"    // generated at build time from Shaders/ldpc_clip_channel.comp by glslc
#include "ldpc_update_cn_spv.hpp"       // generated at build time from Shaders/ldpc_update_cn.comp by glslc
#include "ldpc_update_vn_spv.hpp"       // generated at build time from Shaders/ldpc_update_vn.comp by glslc
#include "ldpc_hard_decision_spv.hpp"   // generated at build time from Shaders/ldpc_hard_decision.comp by glslc

using namespace aff3ct::tools;
using spu::device_interface::compute_api;
using spu::device_interface::memory_type;

// ---------------------------------------------------------------------------
// Single global base graph instance.
// Populated once by ldpc_decoder_init(), read-only from every decode call.
// The GPU-side pointer fields (cn, vn, ...) are set here after upload.
// ---------------------------------------------------------------------------
static Std_5G_base_graph g_bg = {};
static bool              g_initialized = false;

// ---------------------------------------------------------------------------
// Kernel configuration -- mirrors NODE_KERNEL_BLOCK / UNROLL_NODES in the
// CUDA/HIP/SYCL ports (local_size_x = 128 in ldpc_update_{cn,vn}.comp).
// ---------------------------------------------------------------------------
#define NODE_KERNEL_BLOCK 128

static inline uint32_t blocks_for(uint32_t n, uint32_t bs) { return (n + bs - 1) / bs; }

namespace
{
struct ClipPushConstants { uint32_t n; };
struct CnPushConstants { uint32_t Zc; uint32_t cn_stride; uint32_t num_rows; uint32_t first_iter; };
struct VnPushConstants { uint32_t Zc; uint32_t vn_stride; uint32_t num_cols; uint32_t num_rows; };
struct HardDecisionPushConstants { uint32_t block_length; };

// The embedded SPIR-V blobs are stored as raw bytes (see the generated *_spv.hpp headers); Vulkan
// requires the code as a uint32_t* though, so each is copied once, on first use, into a properly
// typed/aligned buffer (a reinterpret_cast from uint8_t* to uint32_t* would be undefined behaviour).
// Mirrors get_matmul_float_spirv_words() in Device/Vulkan/Module/Matmul.cpp.
//
// Their addresses are also half of the dispatch cache's identity (VulkanStream::dispatch_key holds
// `spirv_code` and compares it by pointer, not by content), which is exactly why they are function
// local statics: one blob per shader for the whole process, stable for its whole lifetime.
std::vector<uint32_t>
to_spirv_words(const unsigned char* bytes, size_t size)
{
    std::vector<uint32_t> w(size / sizeof(uint32_t));
    std::memcpy(w.data(), bytes, size);
    return w;
}

const std::vector<uint32_t>& get_clip_spirv_words()
{
    static const std::vector<uint32_t> words =
      to_spirv_words(pipeline_vulkan_shaders::ldpc_clip_channel_spv, pipeline_vulkan_shaders::ldpc_clip_channel_spv_size);
    return words;
}

const std::vector<uint32_t>& get_update_cn_spirv_words()
{
    static const std::vector<uint32_t> words =
      to_spirv_words(pipeline_vulkan_shaders::ldpc_update_cn_spv, pipeline_vulkan_shaders::ldpc_update_cn_spv_size);
    return words;
}

const std::vector<uint32_t>& get_update_vn_spirv_words()
{
    static const std::vector<uint32_t> words =
      to_spirv_words(pipeline_vulkan_shaders::ldpc_update_vn_spv, pipeline_vulkan_shaders::ldpc_update_vn_spv_size);
    return words;
}

const std::vector<uint32_t>& get_hard_decision_spirv_words()
{
    static const std::vector<uint32_t> words =
      to_spirv_words(pipeline_vulkan_shaders::ldpc_hard_decision_spv, pipeline_vulkan_shaders::ldpc_hard_decision_spv_size);
    return words;
}

// ---------------------------------------------------------------------------
// One described chain, plus the key it was described for.
//
// Held by pointer in the context below, never by value in a vector: the descriptors store the
// addresses of the push constant structs that sit right here, so an entry must never be moved
// once built.
// ---------------------------------------------------------------------------
struct ChainEntry
{
    // ---- key: everything a decode call could change under the recorded chain.
    // The base graph derived values (Zc, strides, num_rows/cols) are deliberately absent: they are
    // fixed by ldpc_decoder_init() before the first decode and never move afterwards.
    VkBuffer in_buf   = VK_NULL_HANDLE; // llr_in socket
    VkBuffer out_buf  = VK_NULL_HANDLE; // llr_bits_out socket
    uint32_t num_iter = 0;              // chain length
    uint32_t K        = 0;              // hard decision push constant + group count

    // ---- storage the descriptors point into.
    // VULKAN_dispatch_desc keeps push constants as a `const void*`, so these have to outlive the
    // build and keep the same *bytes* across frames (StreamPU compares their contents, not their
    // address).
    ClipPushConstants         clip_pc{};
    CnPushConstants           cn_pc_first{};
    CnPushConstants           cn_pc_rest{};
    VnPushConstants           vn_pc{};
    HardDecisionPushConstants hd_pc{};

    std::vector<spu::executor::VULKAN_dispatch_desc> chain;

    bool matches(VkBuffer in, VkBuffer out, uint32_t iters, uint32_t k) const
    {
        return in_buf == in && out_buf == out && num_iter == iters && K == k;
    }
};

// Same cap, and for the same reason, as StreamPU's max_recorded_dispatches (Vulkan_device.cpp):
// past this many distinct keys the recordings themselves stop being cached, so keeping more
// descriptions around would buy nothing. One entry is the steady state when the decoder stage runs
// its adaptors in copy mode; a handful when it runs in no-copy mode and the sockets rotate over an
// adaptor's buffer pool.
constexpr size_t max_cached_chains = 8;

// ---------------------------------------------------------------------------
// Per-decoder host-side chain cache.
//
// Derived from sp_vulkan::ThreadContext rather than added to it, so that the shared header stays
// untouched and this whole variant remains one self-contained file. Legitimate because this TU is
// the only place a context is ever allocated (ldpc_decoder_init_context() below) and the only
// place one is ever freed -- there is no deletion through the base pointer, which would be
// undefined with ThreadContext's non-virtual destructor.
//
// No locking anywhere: one context belongs to one Vulkan_decoder, one Vulkan_decoder belongs to
// one replicated decoder module (see Decoder_LDPC_BP_flooding_gpu::clone()), and one of those runs
// on one thread.
// ---------------------------------------------------------------------------
struct CachedContext : sp_vulkan::ThreadContext
{
    std::vector<std::unique_ptr<ChainEntry>> chains;
    size_t last = 0; // entry the previous decode used, tried first

    // Used for a key that arrives once the cache is full, and re-described every time it is. Never
    // evicting a cached entry is deliberate: the sockets rotate *cyclically*, and on a cycle longer
    // than the cache any replacement policy that recycles entries throws away exactly the one
    // needed next and never hits again. Keeping the first max_cached_chains keys pinned instead
    // still hits for those, which is also what StreamPU does with its own recordings.
    std::unique_ptr<ChainEntry> overflow;

    // Diagnostics only (SPU_DEBUG_VULKAN).
    uint64_t builds = 0;
    uint64_t hits   = 0;
};

// ---------------------------------------------------------------------------
// Describe the whole decode as one chain of dispatches, into 'e'.
//
// Identical in content to the loop the original file runs per call -- same dispatches, same order,
// same bindings -- it just writes into a cache entry instead of into a local.
// ---------------------------------------------------------------------------
void build_chain(ChainEntry& e, const CachedContext& ctx, int device_id,
                 VkBuffer in_buf, VkBuffer out_buf, uint32_t num_iter, uint32_t K)
{
    const uint32_t Zc = g_bg.Zc;
    const uint32_t num_vns = g_bg.num_cols * Zc;
    const uint32_t num_cns = g_bg.num_rows * Zc;

    // The working and base graph buffers never move once allocated, but the VkBuffer lookup is a
    // map walk in Devices_manager, so it belongs here rather than on the per-frame path.
    VkBuffer msg_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(ctx.llr_msg_buffer));
    VkBuffer total_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(ctx.llr_total_buffer));
    VkBuffer bg_cn_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.cn)));
    VkBuffer bg_cn_degree_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.cn_degree)));
    VkBuffer bg_vn_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.vn)));
    VkBuffer bg_vn_degree_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.vn_degree)));

    const auto& clip_spirv = get_clip_spirv_words();
    const auto& cn_spirv   = get_update_cn_spirv_words();
    const auto& vn_spirv   = get_update_vn_spirv_words();
    const auto& hd_spirv   = get_hard_decision_spirv_words();

    e.clip_pc     = ClipPushConstants{ num_vns };
    e.cn_pc_first = CnPushConstants{ Zc, g_bg.cn_stride, g_bg.num_rows, 1u };
    e.cn_pc_rest  = CnPushConstants{ Zc, g_bg.cn_stride, g_bg.num_rows, 0u };
    e.vn_pc       = VnPushConstants{ Zc, g_bg.vn_stride, g_bg.num_cols, g_bg.num_rows };
    e.hd_pc       = HardDecisionPushConstants{ K };

    e.chain.clear();
    e.chain.reserve(2 + (size_t)2 * num_iter);

    e.chain.push_back({ clip_spirv.data(), clip_spirv.size() * sizeof(uint32_t),
                        { in_buf, total_buf }, blocks_for(num_vns, 256), 1u,
                        &e.clip_pc, sizeof(e.clip_pc) });

    // Mirrors `float const* llr_total = llr_in;` in the CUDA/HIP/SYCL reference: the first CN
    // update reads the raw (unclipped) channel LLR directly, not clip_channel_kernel's output.
    // The reassignment at the bottom of the loop is a describe-time swap -- each descriptor copied
    // the handle it was pushed with -- so iteration 0 keeps reading in_buf while every later one
    // reads the accumulator the VN update wrote.
    VkBuffer llr_total_current = in_buf;
    for (uint32_t iter = 0; iter < num_iter; ++iter)
    {
        e.chain.push_back({ cn_spirv.data(), cn_spirv.size() * sizeof(uint32_t),
                            { llr_total_current, msg_buf, bg_cn_buf, bg_cn_degree_buf },
                            blocks_for(num_cns, NODE_KERNEL_BLOCK), 1u,
                            (iter == 0 ? &e.cn_pc_first : &e.cn_pc_rest), sizeof(CnPushConstants) });

        e.chain.push_back({ vn_spirv.data(), vn_spirv.size() * sizeof(uint32_t),
                            { msg_buf, in_buf, total_buf, bg_vn_buf, bg_vn_degree_buf },
                            blocks_for(num_vns, NODE_KERNEL_BLOCK), 1u,
                            &e.vn_pc, sizeof(e.vn_pc) });

        llr_total_current = total_buf;
    }

    e.chain.push_back({ hd_spirv.data(), hd_spirv.size() * sizeof(uint32_t),
                        { llr_total_current, out_buf }, blocks_for(K, 256), 1u,
                        &e.hd_pc, sizeof(e.hd_pc) });

    e.in_buf   = in_buf;
    e.out_buf  = out_buf;
    e.num_iter = num_iter;
    e.K        = K;
}

// ---------------------------------------------------------------------------
// The per-frame entry point: hand back the chain for this key, describing it only if no entry
// holds it already.
// ---------------------------------------------------------------------------
const std::vector<spu::executor::VULKAN_dispatch_desc>&
get_chain(CachedContext& ctx, int device_id, VkBuffer in_buf, VkBuffer out_buf, uint32_t num_iter, uint32_t K)
{
    // Fast path: the sockets did not move since the previous frame (always true once the decoder
    // stage's adaptors copy, see --dec-stage-copy in main_gpu.cpp).
    if (ctx.last < ctx.chains.size() && ctx.chains[ctx.last]->matches(in_buf, out_buf, num_iter, K))
    {
        ctx.hits++;
        return ctx.chains[ctx.last]->chain;
    }

    // They did: scan the rest. Rotating over an adaptor's buffer pool lands here, and still hits.
    for (size_t i = 0; i < ctx.chains.size(); ++i)
        if (ctx.chains[i]->matches(in_buf, out_buf, num_iter, K))
        {
            ctx.last = i;
            ctx.hits++;
            return ctx.chains[i]->chain;
        }

    // Genuinely new key: take a fresh entry while there is room, the overflow one afterwards.
    ChainEntry* entry;
    if (ctx.chains.size() < max_cached_chains)
    {
        ctx.chains.push_back(std::unique_ptr<ChainEntry>(new ChainEntry));
        ctx.last = ctx.chains.size() - 1;
        entry = ctx.chains[ctx.last].get();
    }
    else
    {
        if (ctx.overflow == nullptr) ctx.overflow.reset(new ChainEntry);
        // Nothing cached matches, so make sure the fast path above does not claim a stale hit on
        // whatever entry the previous frame happened to use.
        ctx.last = ctx.chains.size();
        entry = ctx.overflow.get();
    }

    ChainEntry& e = *entry;
    build_chain(e, ctx, device_id, in_buf, out_buf, num_iter, K);
    ctx.builds++;

    if (std::getenv("SPU_DEBUG_VULKAN"))
        std::fprintf(stderr,
                     "[ldpc] described a chain of %zu dispatch(es) "
                     "(%llu described, %llu reused, %zu entries cached)\n",
                     e.chain.size(),
                     (unsigned long long)ctx.builds,
                     (unsigned long long)ctx.hits,
                     ctx.chains.size());

    return e.chain;
}
}

// ---------------------------------------------------------------------------
// The implementation itself. Anonymous namespace: nothing outside this file names the type, it is
// reached only through the Vulkan_decoder interface and the factory function at the bottom.
// ---------------------------------------------------------------------------
namespace
{
class Vulkan_decoder_cached : public sp_vulkan::Vulkan_decoder
{
  public:
    explicit Vulkan_decoder_cached(int device_id)
      : sp_vulkan::Vulkan_decoder(device_id)
    {
    }

    ~Vulkan_decoder_cached() override { ldpc_decoder_shutdown(); }

    void ldpc_decoder_init(int K, int N) override;
    void ldpc_decoder_init_context() override;

    uint32_t ldpc_decode(float const* llr_in,
                         uint32_t K,
                         uint32_t num_iter,
                         uint32_t perform_syndrome_check,
                         int* llr_bits_out,
                         spu::device_interface::GpuStream* stream) override;

    void ldpc_decoder_shutdown(void) override;
};
}

// ---------------------------------------------------------------------------
// Upload one table variant's four arrays into GPU memory and populate the
// corresponding pointer + stride fields of the provided base graph struct.
// Called once from ldpc_decoder_init().
// ---------------------------------------------------------------------------
static void upload_tables(Std_5G_base_graph* bg, int dev_id)
{
	// Select the right host-side tables using bg.Bg and bg.index_list
	const uint32_t ils = bg->index_list;

	const uint32_t* h_cn_degree[2][8] = { { BG1_CN_DEGREE_TABLE() },
										   { BG2_CN_DEGREE_TABLE() } };
	const uint32_t* h_vn_degree[2][8] = { { BG1_VN_DEGREE_TABLE() },
										   { BG2_VN_DEGREE_TABLE() } };
	const void* h_cn[2][8] = { { BG1_CN_TABLE() },
										   { BG2_CN_TABLE() } };
	const void* h_vn[2][8] = { { BG1_VN_TABLE() },
										   { BG2_VN_TABLE() } };

	const uint32_t sz_cn_degree[2][8] = { { BG1_CN_DEGREE_TABLE(sizeof) },
										   { BG2_CN_DEGREE_TABLE(sizeof) } };
	const uint32_t sz_vn_degree[2][8] = { { BG1_VN_DEGREE_TABLE(sizeof) },
										   { BG2_VN_DEGREE_TABLE(sizeof) } };
	const uint32_t sz_cn[2][8] = { { BG1_CN_TABLE(sizeof) },
										   { BG2_CN_TABLE(sizeof) } };
	const uint32_t sz_vn[2][8] = { { BG1_VN_TABLE(sizeof) },
										   { BG2_VN_TABLE(sizeof) } };

	const uint32_t b = bg->Bg - 1; // 0-indexed

	auto alloc_and_upload = [dev_id](const void* host_ptr, uint32_t nbytes) -> uint32_t*
	{
		uint8_t* dev_ptr = spu::Devices_manager::allocate_memory(nbytes, { memory_type::DEVICE, compute_api::VULKAN, dev_id, 0 });
		// A memory_type::DEVICE allocation may land on a heap the CPU cannot write to at all, in
		// which case dev_ptr is not a usable destination for memcpy(). upload_memory() picks the
		// right path: a direct copy plus flush when the allocation did turn out to be host-mapped,
		// a staged copy through a temporary host buffer when it did not.
		spu::Devices_manager::upload_vulkan_memory(dev_id, dev_ptr, host_ptr, nbytes);
		return reinterpret_cast<uint32_t*>(dev_ptr);
	};

	bg->cn_degree = alloc_and_upload(h_cn_degree[b][ils], sz_cn_degree[b][ils]);
	bg->vn_degree = alloc_and_upload(h_vn_degree[b][ils], sz_vn_degree[b][ils]);

	bg->cn = alloc_and_upload(h_cn[b][ils], sz_cn[b][ils]);
	bg->cn_stride = sz_cn[b][ils] / (sizeof(bg->cn[0]) * bg->num_rows);

	bg->vn = alloc_and_upload(h_vn[b][ils], sz_vn[b][ils]);
	bg->vn_stride = sz_vn[b][ils] / (sizeof(bg->vn[0]) * bg->num_cols);
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init_context — per-thread working buffer allocation.
// Buffer sizes are derived from the already-resolved g_bg fields.
//
// Allocates a CachedContext, so that the chain built on the first decode has somewhere to live for
// as long as this decoder does. Every clone of the decoder module builds its own (see
// Decoder_LDPC_BP_flooding_gpu::clone()), which is what keeps the cache lock-free: one context per
// decoder, one decoder per thread.
// ---------------------------------------------------------------------------
void Vulkan_decoder_cached::ldpc_decoder_init_context()
{
	auto* ctx = new CachedContext;

	const uint32_t num_vns = g_bg.num_cols * g_bg.Zc;

	uint8_t* msg_ptr = spu::Devices_manager::allocate_memory(
		(size_t)g_bg.num_rows * g_bg.num_cols * g_bg.Zc * sizeof(llr_msg_t),
		{ memory_type::DEVICE, compute_api::VULKAN, this->dev_id, 0 });
	ctx->llr_msg_buffer = reinterpret_cast<llr_msg_t*>(msg_ptr);

	uint8_t* total_ptr = spu::Devices_manager::allocate_memory(
		(size_t)num_vns * sizeof(llr_accumulator_t),
		{ memory_type::DEVICE, compute_api::VULKAN, this->dev_id, 0 });
	ctx->llr_total_buffer = reinterpret_cast<llr_accumulator_t*>(total_ptr);

	this->context = ctx;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init
//
//   1. build_5G_base_graph(K, N)  — resolves Bg, Zc, index_list,
//                                   num_rows, num_cols, num_edges
//   2. upload_tables(g_bg)        — fills cn/vn/cn_degree/vn_degree + strides
//   3. ldpc_decoder_init_context  — allocates per-thread GPU working buffers
// ---------------------------------------------------------------------------
void
Vulkan_decoder_cached::ldpc_decoder_init(int K, int N)
{
	// Delegate all BG selection and Zc resolution to your helper.
	// Throws std::invalid_argument on bad (K, N) — let it propagate.
	g_bg = build_5G_base_graph(K, N);

	printf("LDPC init: K=%d N=%d -> BG%u Zc=%u ils=%u rows=%u cols=%u (cached dispatch chain)\n",
		K, N, g_bg.Bg, g_bg.Zc, g_bg.index_list,
		g_bg.num_rows, g_bg.num_cols);

	// Upload only the one (Bg, index_list) table variant we need
	upload_tables(&g_bg, this->dev_id);
	ldpc_decoder_init_context();

	g_initialized = true;
}

// ---------------------------------------------------------------------------
// ldpc_decode
//
// Steady-state path: check four scalars, then submit the chain that is already described and
// already recorded. Everything else only runs on the first frame (or after the sockets moved).
// ---------------------------------------------------------------------------
uint32_t
Vulkan_decoder_cached::ldpc_decode(
	float const*   llr_in,            // g_bg.num_cols * g_bg.Zc bytes
	uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
	uint32_t       num_iter,
	uint32_t       perform_syndrome_check,
	int* llr_bits_out,
	spu::device_interface::GpuStream* stream)
{
	auto vulkan_stream = static_cast<spu::sp_vulkan::VulkanStream*>(stream);
	const int device_id = this->dev_id;

	auto& ctx = *static_cast<CachedContext*>(this->context);

	// The only two things the framework can hand us differently from one frame to the next. Both
	// are baked into the recorded descriptor sets, so a change here invalidates the description
	// as much as it invalidates StreamPU's recording.
	VkBuffer in_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<float*>(llr_in)));
	VkBuffer out_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(llr_bits_out));

	const auto& chain = get_chain(ctx, device_id, in_buf, out_buf, num_iter, K);

	// Left at the executor's default mode, which is CACHED unless SPU_VULKAN_DISPATCH_MODE says
	// otherwise -- so the record-every-frame behaviour stays measurable against this one without a
	// rebuild. Constructing the executor per call is free: the pipeline cache is a static keyed on
	// the SPIR-V blob, and the recorded command buffer belongs to the stream, not to the executor.
	spu::executor::VULKAN_executor exec(vulkan_stream->device());
	exec.set_stream(vulkan_stream);

	// --gpu-dispatch. CACHED reuses the recorded VkCommandBuffer for this exact chain; ONE_SHOT
	// re-records it every frame. Set explicitly rather than left to the executor's environment
	// default, so one switch drives this and the CUDA graph path together.
	exec.set_dispatch_mode(gpu_dispatch::get() == gpu_dispatch::mode::CACHED
	                         ? spu::executor::vk_dispatch_mode::CACHED
	                         : spu::executor::vk_dispatch_mode::ONE_SHOT);

	// Profiling is a per-run choice (--dec-vk-profile / SPU_VULKAN_PROFILE), so force the
	// executor's own flag to match ours instead of letting the two disagree: return_profiling_times()
	// throws when nothing was recorded, and would silently collect behind our back if it were on
	// here and off there.
	const bool profiled = gpu_prof::enabled();
	exec.set_profiling(profiled);

	// No flush of llr_in here: it is written by the upstream native task (the puncturer's
	// depuncture) and read by the first dispatch, so it is only needed when allocate_memory() fell
	// back to a cached, non-coherent memory type (see VULKAN_device::allocate_memory /
	// flush_memory -- it is a no-op on coherent allocations). Restore it if a platform ever
	// reports corrupted LLRs.

	{
		// Every VulkanStream shares one VkQueue; vkQueueSubmit() needs external synchronisation.
		std::lock_guard<std::mutex> submit_lock(sp_vulkan::submit_mutex());
		exec.dispatch_chain_and_wait(chain, profiled);
	}

	// One record per launch, and the chain is one launch, so this is a single (cpu, gpu) pair in
	// microseconds. Folded into this decoder's totals; main_gpu.cpp prints them once the pipeline
	// has stopped.
	if (profiled) this->prof.add(exec.return_profiling_times());

	// No invalidate of llr_bits_out either. The hard-decision dispatch wrote that buffer (this
	// task's V_K socket) and the downstream native tasks (the monitor's check_errors, the sink's
	// send_count) read it from the CPU; dispatch_chain_and_wait() already fence-waited, so the
	// write itself has landed. The invalidate only mattered on hardware where the allocation is
	// cached but not coherent, where the CPU's view can be stale.

	return num_iter - 1;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_shutdown
//
// Frees the host-side chain and the context holding it. The GPU allocations behind
// llr_msg_buffer/llr_total_buffer belong to Devices_manager and go with the device, and the
// recorded command buffers and descriptor pools belong to the VulkanStream and go with it.
// Deleting as CachedContext*, never as ThreadContext*: the base has no virtual destructor.
// ---------------------------------------------------------------------------
void Vulkan_decoder_cached::ldpc_decoder_shutdown()
{
	delete static_cast<CachedContext*>(this->context);
	this->context = nullptr;
}

// ---------------------------------------------------------------------------
// Declared in Decoder_LDPC_vulkan_kernel.hpp, called only by Vulkan_decoder::create().
// ---------------------------------------------------------------------------
sp_vulkan::Vulkan_decoder*
sp_vulkan::make_decoder_chain_cached(int device_id)
{
	return new Vulkan_decoder_cached(device_id);
}

#endif // DECODER_VULKAN
