/*
Vulkan port of the CUDA/HIP/SYCL LDPC 5G decoder (see src/cuda/Decoder_LDPC_cuda_kernel.cu,
src/hip/Decoder_LDPC_hip_kernel.hip and src/sycl/Decoder_LDPC_sycl_kernel.cpp).

Variant using aff3ct::tools::build_5G_base_graph() for base graph selection.
The Std_5G_base_graph struct subsumes the original BaseGraph struct entirely --
its gpu-side pointer fields (cn, vn, cn_degree, vn_degree, cn_stride, vn_stride)
are populated once during init and reused on every decode call.

Unlike CUDA/HIP/SYCL, Vulkan has no native GPU pointer: every buffer is addressed
through a VkBuffer bound into a descriptor set, and the check/variable-node update
kernels each need more storage buffers (4 and 5 respectively) than a single fixed-in/
fixed-out dispatch can express, so the whole decode iteration is built as one chain of
spu::executor::VULKAN_dispatch_desc entries and submitted with a single
dispatch_chain_and_wait() call (one submit + one fence wait for the whole decode,
instead of one round trip per kernel launch).
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include <vulkan/vulkan.h>

#include "Vulkan/Decoder_LDPC_vulkan_kernel.hpp"
#include "Vulkan/vulkan_submit_lock.hpp"
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

//Init the GPU handler
sp_vulkan::Vulkan_decoder::Vulkan_decoder(int device_id)
	: dev_id(device_id), context(nullptr)
{
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init_context — per-thread working buffer allocation.
// Buffer sizes are derived from the already-resolved g_bg fields.
// ---------------------------------------------------------------------------
void sp_vulkan::Vulkan_decoder::ldpc_decoder_init_context()
{
	this->context = new sp_vulkan::ThreadContext;

	const uint32_t num_vns = g_bg.num_cols * g_bg.Zc;

	uint8_t* msg_ptr = spu::Devices_manager::allocate_memory(
		(size_t)g_bg.num_rows * g_bg.num_cols * g_bg.Zc * sizeof(llr_msg_t),
		{ memory_type::DEVICE, compute_api::VULKAN, this->dev_id, 0 });
	this->context->llr_msg_buffer = reinterpret_cast<llr_msg_t*>(msg_ptr);

	uint8_t* total_ptr = spu::Devices_manager::allocate_memory(
		(size_t)num_vns * sizeof(llr_accumulator_t),
		{ memory_type::DEVICE, compute_api::VULKAN, this->dev_id, 0 });
	this->context->llr_total_buffer = reinterpret_cast<llr_accumulator_t*>(total_ptr);
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init
//
// Replaces the original [2][8] upload loop with:
//   1. build_5G_base_graph(K, N)  — your helper resolves Bg, Zc, index_list,
//                                   num_rows, num_cols, num_edges
//   2. upload_tables(g_bg)        — fills cn/vn/cn_degree/vn_degree + strides
//   3. ldpc_decoder_init_context  — allocates per-thread GPU working buffers
// ---------------------------------------------------------------------------
void
sp_vulkan::Vulkan_decoder::ldpc_decoder_init(int K, int N)
{
	// Delegate all BG selection and Zc resolution to your helper.
	// Throws std::invalid_argument on bad (K, N) — let it propagate.
	g_bg = build_5G_base_graph(K, N);

	printf("LDPC init: K=%d N=%d -> BG%u Zc=%u ils=%u rows=%u cols=%u\n",
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
// Parameters simplified vs the original:
//   - BG and Z are gone (read from g_bg)
//   - block_length kept as K (the caller's original K, not K_LDPC)
// ---------------------------------------------------------------------------
uint32_t
sp_vulkan::Vulkan_decoder::ldpc_decode(
	float const*   llr_in,            // g_bg.num_cols * g_bg.Zc bytes
	uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
	uint32_t       num_iter,
	uint32_t       perform_syndrome_check,
	int* llr_bits_out,
	spu::device_interface::GpuStream* stream)
{
	auto vulkan_stream = static_cast<spu::sp_vulkan::VulkanStream*>(stream);
	const int device_id = this->dev_id;

	const uint32_t Zc = g_bg.Zc;
	const uint32_t num_vns = g_bg.num_cols * Zc;
	const uint32_t num_cns = g_bg.num_rows * Zc;

	VkBuffer in_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<float*>(llr_in)));
	VkBuffer out_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(llr_bits_out));
	VkBuffer msg_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(this->context->llr_msg_buffer));
	VkBuffer total_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(this->context->llr_total_buffer));
	VkBuffer bg_cn_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.cn)));
	VkBuffer bg_cn_degree_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.cn_degree)));
	VkBuffer bg_vn_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.vn)));
	VkBuffer bg_vn_degree_buf = spu::Devices_manager::get_vulkan_buffer(device_id, reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(g_bg.vn_degree)));

	const auto& clip_spirv = get_clip_spirv_words();
	const auto& cn_spirv   = get_update_cn_spirv_words();
	const auto& vn_spirv   = get_update_vn_spirv_words();
	const auto& hd_spirv   = get_hard_decision_spirv_words();

	ClipPushConstants clip_pc{ num_vns };
	CnPushConstants cn_pc_first{ Zc, g_bg.cn_stride, g_bg.num_rows, 1u };
	CnPushConstants cn_pc_rest{ Zc, g_bg.cn_stride, g_bg.num_rows, 0u };
	VnPushConstants vn_pc{ Zc, g_bg.vn_stride, g_bg.num_cols, g_bg.num_rows };
	HardDecisionPushConstants hd_pc{ K };

	std::vector<spu::executor::VULKAN_dispatch_desc> chain;
	chain.reserve(2 + (size_t)2 * num_iter);

	chain.push_back({ clip_spirv.data(), clip_spirv.size() * sizeof(uint32_t),
	                   { in_buf, total_buf }, blocks_for(num_vns, 256), 1u,
	                   &clip_pc, sizeof(clip_pc) });

	// Mirrors `float const* llr_total = llr_in;` in the CUDA/HIP/SYCL reference: the first CN
	// update reads the raw (unclipped) channel LLR directly, not clip_channel_kernel's output.
	VkBuffer llr_total_current = in_buf;
	for (uint32_t iter = 0; iter < num_iter; ++iter)
	{
		chain.push_back({ cn_spirv.data(), cn_spirv.size() * sizeof(uint32_t),
		                   { llr_total_current, msg_buf, bg_cn_buf, bg_cn_degree_buf },
		                   blocks_for(num_cns, NODE_KERNEL_BLOCK), 1u,
		                   (iter == 0 ? &cn_pc_first : &cn_pc_rest), sizeof(CnPushConstants) });

		chain.push_back({ vn_spirv.data(), vn_spirv.size() * sizeof(uint32_t),
		                   { msg_buf, in_buf, total_buf, bg_vn_buf, bg_vn_degree_buf },
		                   blocks_for(num_vns, NODE_KERNEL_BLOCK), 1u,
		                   &vn_pc, sizeof(vn_pc) });

		llr_total_current = total_buf;
	}

	chain.push_back({ hd_spirv.data(), hd_spirv.size() * sizeof(uint32_t),
	                   { llr_total_current, out_buf }, blocks_for(K, 256), 1u,
	                   &hd_pc, sizeof(hd_pc) });

	spu::executor::VULKAN_executor exec(vulkan_stream->device());
	exec.set_stream(vulkan_stream);

	// llr_in was just written by the upstream native task feeding this task's Y_N socket (the
	// puncturer's depuncture); make sure the GPU sees it before the first dispatch reads it
	// (see VULKAN_device::flush_memory -- a no-op when the allocation landed on coherent memory).
	spu::Devices_manager::flush_vulkan_memory(device_id, reinterpret_cast<uint8_t*>(const_cast<float*>(llr_in)));

	{
		// Every VulkanStream shares one VkQueue; vkQueueSubmit() needs external synchronisation.
		std::lock_guard<std::mutex> submit_lock(sp_vulkan::submit_mutex());
		exec.dispatch_chain_and_wait(chain);
	}

	// The hard-decision dispatch just wrote llr_bits_out (this task's V_K socket), which the
	// downstream native tasks (the monitor's check_errors and the sink's send_count) read directly
	// from the CPU once this call returns. dispatch_chain_and_wait() already fence-waited, so the
	// write itself is done, but on hardware where allocate_memory() had to fall back to a
	// cached-but-non-coherent memory type (see VULKAN_device::allocate_memory) the CPU's view of it
	// can still be stale until invalidated.
	spu::Devices_manager::invalidate_vulkan_memory(device_id, reinterpret_cast<uint8_t*>(llr_bits_out));

	return num_iter - 1;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_shutdown
// ---------------------------------------------------------------------------
void sp_vulkan::Vulkan_decoder::ldpc_decoder_shutdown() {}
