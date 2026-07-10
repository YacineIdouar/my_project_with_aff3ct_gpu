/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

SYCL port of the CUDA/HIP LDPC 5G decoder (see src/cuda/Decoder_LDPC_cuda_kernel.cu
and src/hip/Decoder_LDPC_hip_kernel.hip).

Variant using aff3ct::tools::build_5G_base_graph() for base graph selection.
The Std_5G_base_graph struct subsumes the original BaseGraph struct entirely —
its gpu-side pointer fields (cn, vn, cn_degree, vn_degree, cn_stride, vn_stride)
are populated once during init and reused on every decode call.

Threading model: the CUDA/HIP kernels are launched with a 2D block
(NODE_KERNEL_BLOCK, UNROLL_NODES) and a 1D grid (only blockIdx.x is used), so
threadIdx.y never varies across blocks — it only unrolls the per-node loop
inside a single block. This is reproduced here with a 2D nd_range whose global
range in dimension 1 equals its local range (UNROLL_NODES), i.e. get_global_id(0)
plays the role of CUDA's "tid" and get_local_id(1) plays the role of threadIdx.y.
*/

#include <sycl/sycl.hpp>
#include <cstdio>
#include <unistd.h>
#include <stdint.h>
#include <iostream>

#include "Sycl/Decoder_LDPC_sycl_kernel.hpp"
#include "Device/Sycl/Sycl_executor.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg1.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg2.hpp"
#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"

using namespace aff3ct::tools;

// ---------------------------------------------------------------------------
// Data types (unchanged)
// ---------------------------------------------------------------------------

static constexpr float MAX_CHANNEL_LLR = 128.0f;
#define OMS_OFFSET 0.5f
// ---------------------------------------------------------------------------
// Single global base graph instance.
// Populated once by ldpc_decoder_init(), read-only from every decode call.
// The GPU-side pointer fields (cn, vn, ...) are set here after upload.
// ---------------------------------------------------------------------------
static Std_5G_base_graph g_bg = {};
static bool              g_initialized = false;

// ---------------------------------------------------------------------------
// Kernel configuration
// ---------------------------------------------------------------------------
#define NODE_KERNEL_BLOCK 128
#define UNROLL_NODES      1

static inline uint32_t blocks_for(uint32_t n, uint32_t bs) {
	return (n + bs - 1) / bs;
}

static void clip_channel_kernel(sycl::queue& q,
	float const* in,
	llr_accumulator_t* out,
	uint32_t n)
{
	q.submit([&](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			uint32_t tid = idx[0];
			out[tid] = sycl::fmax(-MAX_CHANNEL_LLR, sycl::fmin(MAX_CHANNEL_LLR, in[tid]));
		});
	});
}
// ---------------------------------------------------------------------------
// update_cn_kernel
// Uses Std_5G_base_graph fields directly: num_rows, cn, cn_degree, cn_stride.
// Zc passed as a plain uint32_t (resolved once, stored in g_bg.Zc).
// ---------------------------------------------------------------------------
static void update_cn_kernel(
	sycl::queue& q,
	llr_accumulator_t const* llr_total,
	llr_msg_t* llr_msg,
	uint32_t Zc,
	uint32_t const* bg_cn,
	uint32_t const* bg_cn_degree,
	uint32_t cn_stride,
	uint32_t num_rows,
	bool first_iter)
{
	const uint32_t num_cns = num_rows * Zc;
	const uint32_t gx = blocks_for(num_cns, NODE_KERNEL_BLOCK) * NODE_KERNEL_BLOCK;
	sycl::nd_range<2> ndr(sycl::range<2>(gx, UNROLL_NODES), sycl::range<2>(NODE_KERNEL_BLOCK, UNROLL_NODES));

	q.submit([&](sycl::handler& h) {
		h.parallel_for(ndr, [=](sycl::nd_item<2> item) {
			const uint32_t tid = item.get_global_id(0);
			const uint32_t ty = item.get_local_id(1);
			const uint32_t i = tid % Zc;
			const uint32_t idx_row = tid / Zc;

			if (idx_row >= num_rows) return;

			const uint32_t cn_degree = bg_cn_degree[idx_row];
			uint32_t const* check_nodes = &bg_cn[idx_row * cn_stride];

			float      min_1 = 1e38f;
			float      min_2 = 1e38f;
			int      idx_min = -1;
			int      node_sign = 1;
			uint32_t msg_signs = 0;

			for (uint32_t ii = ty; ii < cn_degree; ii += UNROLL_NODES)
			{
				uint32_t cn = check_nodes[ii];
				uint32_t idx_col = cn & 0xffffu;
				uint32_t s = cn >> 16;
				uint32_t msg_offset = idx_row + idx_col * num_rows;
				uint32_t msg_idx = msg_offset * Zc + i;

				float t = llr_total[idx_col * Zc + (i + s) % Zc];

				if (!first_iter)
					t -= llr_msg[msg_idx];

				int sign = (t >= 0 ? 1 : -1);
				node_sign *= sign;
				msg_signs |= (t < 0) << ii; //

				float t_abs = sycl::fabs(t);
				if (t_abs < min_1) { min_2 = min_1; min_1 = t_abs; idx_min = msg_idx; }
				else if (t_abs < min_2) min_2 = t_abs;
			}

			min_1 = sycl::fmax(0.f, min_1 - OMS_OFFSET);
			min_2 = sycl::fmax(0.f, min_2 - OMS_OFFSET);
			// END marker-cnp-damping

			for (uint32_t ii = ty; ii < cn_degree; ii += UNROLL_NODES) {
				uint32_t cn = check_nodes[ii];

				// see packing layout above
				uint32_t idx_col = cn & 0xffffu; // note: little endian
				uint32_t msg_offset = idx_row + idx_col * num_rows;

				uint32_t msg_idx = msg_offset * Zc + i;
				float min_val;
				if (msg_idx == idx_min)
					min_val = min_2;
				else
					min_val = min_1;

				int msg_sign = (msg_signs >> ii) & 0x1 ? -1 : 1;

				// and update outgoing msg including sign
				llr_msg[msg_idx] = llr_msg_t(min_val * node_sign * msg_sign);
			}
		});
	});
}

// ---------------------------------------------------------------------------
// update_vn_kernel
// ---------------------------------------------------------------------------
static void update_vn_kernel(
	sycl::queue& q,
	llr_msg_t const* llr_msg,
	float const* llr_ch,
	llr_accumulator_t* llr_total,
	uint32_t Zc,
	uint32_t const* bg_vn,
	uint32_t const* bg_vn_degree,
	uint32_t vn_stride,
	uint32_t num_cols,
	uint32_t num_rows)
{
	const uint32_t num_vns = num_cols * Zc;
	const uint32_t gx = blocks_for(num_vns, NODE_KERNEL_BLOCK) * NODE_KERNEL_BLOCK;
	sycl::nd_range<2> ndr(sycl::range<2>(gx, UNROLL_NODES), sycl::range<2>(NODE_KERNEL_BLOCK, UNROLL_NODES));

	q.submit([&](sycl::handler& h) {
		h.parallel_for(ndr, [=](sycl::nd_item<2> item) {
			const uint32_t tid = item.get_global_id(0);
			const uint32_t ty = item.get_local_id(1);
			const uint32_t i = tid % Zc;
			const uint32_t idx_col = tid / Zc;

			if (idx_col >= num_cols) return;

			const uint32_t vn_degree = bg_vn_degree[idx_col];
			uint32_t const* variable_nodes = &bg_vn[idx_col * vn_stride];

			float msg_sum = 0;
			for (uint32_t j = ty; j < vn_degree; j += UNROLL_NODES)
			{
				uint32_t vn = variable_nodes[j];
				uint32_t idx_row = vn & 0xffffu;
				uint32_t s = vn >> 16;
				uint32_t msg_offset = idx_row + idx_col * num_rows;
				uint32_t msg_idx = msg_offset * Zc + (i - s + (Zc << 8)) % Zc;
				msg_sum += llr_msg[msg_idx];
			}
			if (ty == 0)
			{
				msg_sum += llr_ch[idx_col * Zc + i];
			}

			//msg_sum = sycl::fmax(-MAX_LLR_ACCUMULATOR_VALUE, sycl::fmin(MAX_LLR_ACCUMULATOR_VALUE, msg_sum));

			llr_total[idx_col * Zc + i] = llr_accumulator_t(msg_sum);
		});
	});
}

// ---------------------------------------------------------------------------
// Hard Decision
// ---------------------------------------------------------------------------

static void hard_decision_kernel(
	sycl::queue& q,
	llr_accumulator_t const* llr_total,
	int* bits,
	uint32_t block_length)
{
	q.submit([&](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(block_length), [=](sycl::id<1> idx) {
			uint32_t tid = idx[0];
			bits[tid] = (llr_total[tid] < 0) ? 1 : 0;
		});
	});
}

// ---------------------------------------------------------------------------
// Upload one table variant's four arrays into GPU memory and populate the
// corresponding pointer + stride fields of the provided base graph struct.
// Called once from ldpc_decoder_init().
// ---------------------------------------------------------------------------
static void upload_tables(Std_5G_base_graph* bg, sycl::queue& q)
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

	bg->cn_degree = sycl::malloc_device<uint32_t>(sz_cn_degree[b][ils], q);
	q.memcpy(const_cast<uint32_t*>(bg->cn_degree), h_cn_degree[b][ils], sz_cn_degree[b][ils]).wait();

	bg->vn_degree = sycl::malloc_device<uint32_t>(sz_vn_degree[b][ils], q);
	q.memcpy(const_cast<uint32_t*>(bg->vn_degree), h_vn_degree[b][ils], sz_vn_degree[b][ils]).wait();

	bg->cn = sycl::malloc_device<uint32_t>(sz_cn[b][ils], q);
	q.memcpy(const_cast<uint32_t*>(bg->cn), h_cn[b][ils], sz_cn[b][ils]).wait();
	bg->cn_stride = sz_cn[b][ils] / (sizeof(bg->cn[0]) * bg->num_rows);

	bg->vn = sycl::malloc_device<uint32_t>(sz_vn[b][ils], q);
	q.memcpy(const_cast<uint32_t*>(bg->vn), h_vn[b][ils], sz_vn[b][ils]).wait();
	bg->vn_stride = sz_vn[b][ils] / (sizeof(bg->vn[0]) * bg->num_cols);
}

//Init the GPU handler
sp_sycl::Sycl_decoder::Sycl_decoder(int device_id, int platform_id)
	: dev_id(device_id), platform_id(platform_id), executor(new spu::executor::SYCL_executor(dev_id, platform_id)), context(nullptr)
{
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init_context — per-thread working buffer allocation.
// Buffer sizes are derived from the already-resolved g_bg fields.
// ---------------------------------------------------------------------------
void sp_sycl::Sycl_decoder::ldpc_decoder_init_context()
{
	this->context = new sp_sycl::ThreadContext;

	sycl::queue q = this->executor->get_exec_queue();

	const uint32_t num_vns = g_bg.num_cols * g_bg.Zc;
	const uint32_t num_cns = g_bg.num_rows * g_bg.Zc;

	this->context->llr_in_buffer = sycl::malloc_host<float>(num_vns, q);

	this->context->llr_bits_out_buffer = sycl::malloc_host<int>(g_bg.K_LDPC, q);

	this->context->syndrome_buffer = sycl::malloc_host<uint32_t>(num_cns / 32, q);

	this->context->llr_msg_buffer = sycl::malloc_device<llr_msg_t>(
		(size_t)g_bg.num_rows * g_bg.num_cols * g_bg.Zc, q);

	this->context->llr_total_buffer = sycl::malloc_device<llr_accumulator_t>(num_vns, q);
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
sp_sycl::Sycl_decoder::ldpc_decoder_init(int K, int N)
{
	// Delegate all BG selection and Zc resolution to your helper.
	// Throws std::invalid_argument on bad (K, N) — let it propagate.
	g_bg = build_5G_base_graph(K, N);

	printf("LDPC init: K=%d N=%d -> BG%u Zc=%u ils=%u rows=%u cols=%u\n",
		K, N, g_bg.Bg, g_bg.Zc, g_bg.index_list,
		g_bg.num_rows, g_bg.num_cols);

	sycl::queue q = this->executor->get_exec_queue();

	// Upload only the one (Bg, index_list) table variant we need
	upload_tables(&g_bg, q);
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
sp_sycl::Sycl_decoder::ldpc_decode(
	float const*   llr_in,            // g_bg.num_cols * g_bg.Zc bytes
	uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
	uint32_t       num_iter,
	uint32_t       perform_syndrome_check,
	int* llr_bits_out,
	spu::device_interface::GpuStream* stream)
{
	sycl::queue& native_stream = stream->cast<sycl::queue&>();

	const uint32_t Zc = g_bg.Zc;
	const uint32_t num_vns = g_bg.num_cols * Zc;
	const uint32_t num_cns = g_bg.num_rows * Zc;

	clip_channel_kernel(native_stream, llr_in, this->context->llr_total_buffer, num_vns);

	float const* llr_total = llr_in;

	for (uint32_t iter = 0; iter < num_iter; ++iter)
	{
		update_cn_kernel(native_stream, llr_total, this->context->llr_msg_buffer, Zc, g_bg.cn, g_bg.cn_degree, g_bg.cn_stride, g_bg.num_rows, iter == 0);
		update_vn_kernel(native_stream, this->context->llr_msg_buffer, llr_in, this->context->llr_total_buffer, Zc, g_bg.vn, g_bg.vn_degree, g_bg.vn_stride, g_bg.num_cols, g_bg.num_rows);
		llr_total = this->context->llr_total_buffer;
	}
	hard_decision_kernel(native_stream, llr_total, llr_bits_out, K);

	native_stream.wait();

	return num_iter - 1;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_shutdown
// ---------------------------------------------------------------------------
void sp_sycl::Sycl_decoder::ldpc_decoder_shutdown() {}
