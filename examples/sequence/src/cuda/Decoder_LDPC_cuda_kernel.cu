/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Variant using aff3ct::tools::build_5G_base_graph() for base graph selection.
The Std_5G_base_graph struct subsumes the original BaseGraph struct entirely —
its gpu-side pointer fields (cn, vn, cn_degree, vn_degree, cn_stride, vn_stride)
are populated once during init and reused on every decode call.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>
#include <stdint.h>
#include <iostream>

#include "Decoder_LDPC_cuda_kernel.hpp"
#include "ldpc_tables_bg1.hpp"
#include "ldpc_tables_bg2.hpp"
#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"

using namespace aff3ct::tools;

// ---------------------------------------------------------------------------
// Data types (unchanged)
// ---------------------------------------------------------------------------
static constexpr int MAX_LLR_ACCUMULATOR_VALUE = 127;
typedef int8_t llr_accumulator_t;
static constexpr int MAX_LLR_MSG_VALUE = 127;
typedef int8_t llr_msg_t;

#define APPLY_DAMPING_INT(x) ((x)*3/4)

// ---------------------------------------------------------------------------
// Single global base graph instance.
// Populated once by ldpc_decoder_init(), read-only from every decode call.
// The GPU-side pointer fields (cn, vn, ...) are set here after upload.
// ---------------------------------------------------------------------------
static Std_5G_base_graph g_bg = {};
static bool              g_initialized = false;

// ---------------------------------------------------------------------------
// Thread context (one per OS thread)
// ---------------------------------------------------------------------------
struct ThreadContext {
	cudaStream_t       stream = 0;
	int8_t* llr_in_buffer = nullptr;
	int* llr_bits_out_buffer = nullptr;
	uint32_t* syndrome_buffer = nullptr;
	llr_msg_t* llr_msg_buffer = nullptr;
	llr_accumulator_t* llr_total_buffer = nullptr;
};

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e) printf("CUDA error %d: %s at %s:%d\n", (int)_e, \
                   cudaGetErrorString(_e), __FILE__, __LINE__); \
} while(0)

// ---------------------------------------------------------------------------
// Kernel configuration
// ---------------------------------------------------------------------------
#define NODE_KERNEL_BLOCK 128
#define UNROLL_NODES      1

inline __host__ __device__ uint32_t blocks_for(uint32_t n, uint32_t bs) {
	return (n + bs - 1) / bs;
}

// ---------------------------------------------------------------------------
// update_cn_kernel
// Uses Std_5G_base_graph fields directly: num_rows, cn, cn_degree, cn_stride.
// Zc passed as a plain uint32_t (resolved once, stored in g_bg.Zc).
// ---------------------------------------------------------------------------
__launch_bounds__(UNROLL_NODES* NODE_KERNEL_BLOCK, 3)
static __global__ void update_cn_kernel(
	llr_accumulator_t const* llr_total,
	llr_msg_t* llr_msg,
	uint32_t Zc,
	uint32_t const* bg_cn,
	uint32_t const* bg_cn_degree,
	uint32_t cn_stride,
	uint32_t num_rows,
	bool first_iter)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t i = tid % Zc;
	const uint32_t idx_row = tid / Zc;

	if (idx_row >= num_rows) return;

	const uint32_t cn_degree = bg_cn_degree[idx_row];
	uint32_t const* check_nodes = &bg_cn[idx_row * cn_stride];

	int      min_1 = INT_MAX;
	int      min_2 = INT_MAX;
	int      idx_min = -1;
	int      node_sign = 1;
	uint32_t msg_signs = 0;

#if UNROLL_NODES > 1
	__shared__ int      mins1[UNROLL_NODES][NODE_KERNEL_BLOCK + 1];
	__shared__ int      mins2[UNROLL_NODES][NODE_KERNEL_BLOCK + 1];
	__shared__ int      idx_mins[UNROLL_NODES][NODE_KERNEL_BLOCK + 1];
	__shared__ uint32_t signs[UNROLL_NODES][NODE_KERNEL_BLOCK + 1];
	mins1[threadIdx.y][threadIdx.x] = INT_MAX;
	mins2[threadIdx.y][threadIdx.x] = INT_MAX;
	signs[threadIdx.y][threadIdx.x] = 0;
#endif

	__syncwarp();

	for (uint32_t ii = threadIdx.y; ii < cn_degree; ii += UNROLL_NODES) {
		uint32_t cn = check_nodes[ii];
		uint32_t idx_col = cn & 0xffffu;
		uint32_t s = cn >> 16;
		uint32_t msg_offset = idx_row + idx_col * num_rows;
		uint32_t msg_idx = msg_offset * Zc + i;

		int t = llr_total[idx_col * Zc + (i + s) % Zc];
		if (!first_iter)
			t -= __ldg(&llr_msg[msg_idx]);

		int sign = (t >= 0 ? 1 : -1);
		node_sign *= sign;
		msg_signs |= (t < 0) << ii; //

		int t_abs = abs(t);
		if (t_abs < min_1) { min_2 = min_1; min_1 = t_abs; idx_min = msg_idx; }
		else if (t_abs < min_2) min_2 = t_abs;
	}

#if UNROLL_NODES > 1
	mins1[threadIdx.y][threadIdx.x] = min_1;
	mins2[threadIdx.y][threadIdx.x] = min_2;
	idx_mins[threadIdx.y][threadIdx.x] = idx_min;
	signs[threadIdx.y][threadIdx.x] = msg_signs;

	__syncthreads();

	if (threadIdx.y == 0) {
		min_1 = INT_MAX; min_2 = INT_MAX; idx_min = -1; msg_signs = 0;
		for (int k = 0; k < UNROLL_NODES; ++k) {
			int t_abs = mins1[k][threadIdx.x];
			if (t_abs < min_1) { min_2 = min_1; min_1 = t_abs; idx_min = idx_mins[k][threadIdx.x]; }
			else if (t_abs < min_2) min_2 = t_abs;
			min_2 = min(min_2, mins2[k][threadIdx.x]);
			msg_signs |= signs[k][threadIdx.x];
		}
		mins1[0][threadIdx.x] = min_1;
		mins2[0][threadIdx.x] = min_2;
		idx_mins[0][threadIdx.x] = idx_min;
		signs[0][threadIdx.x] = msg_signs;
	}

	__syncthreads();

	min_1 = mins1[0][threadIdx.x];
	min_2 = mins2[0][threadIdx.x];
	idx_min = idx_mins[0][threadIdx.x];
	msg_signs = signs[0][threadIdx.x];
#endif

	min_1 = APPLY_DAMPING_INT(min_1); // min_1 * DAMPING_FACTOR, e.g. *3/4
	min_2 = APPLY_DAMPING_INT(min_2); // min_2 * DAMPING_FACTOR, e.g. *3/4
	// END marker-cnp-damping

	// clip msg magnitudes to MAX_LLR_VALUE
	min_1 = min(max(min_1, -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);
	min_2 = min(max(min_2, -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);
	// END marker-vnp-clipping

	__syncwarp();

	if (idx_row < num_rows) {
		for (uint32_t ii = threadIdx.y; ii < cn_degree; ii += UNROLL_NODES) {
			uint32_t cn = check_nodes[ii];

			// see packing layout above
			uint32_t idx_col = cn & 0xffffu; // note: little endian
			uint32_t msg_offset = idx_row + idx_col * num_rows;

			uint32_t msg_idx = msg_offset * Zc + i;
			int min_val;
			if (msg_idx == idx_min)
				min_val = min_2;
			else
				min_val = min_1;

			int msg_sign = (msg_signs >> ii) & 0x1 ? -1 : 1;

			// and update outgoing msg including sign
			llr_msg[msg_idx] = llr_msg_t(min_val * node_sign * msg_sign);
		}
	}

}

// ---------------------------------------------------------------------------
// update_vn_kernel
// ---------------------------------------------------------------------------
__launch_bounds__(UNROLL_NODES* NODE_KERNEL_BLOCK, 3)
static __global__ void update_vn_kernel(
	llr_msg_t const*  llr_msg,
	int8_t const*  llr_ch,
	llr_accumulator_t*  llr_total,
	uint32_t Zc,
	uint32_t const*  bg_vn,
	uint32_t const*  bg_vn_degree,
	uint32_t vn_stride,
	uint32_t num_cols,
	uint32_t num_rows)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t i = tid % Zc;
	const uint32_t idx_col = tid / Zc;

	if (idx_col >= num_cols) return;

	const uint32_t vn_degree = bg_vn_degree[idx_col];
	uint32_t const* variable_nodes = &bg_vn[idx_col * vn_stride];

#if UNROLL_NODES > 1
	__shared__ int msg_sums[UNROLL_NODES][NODE_KERNEL_BLOCK + 1];
	msg_sums[threadIdx.y][threadIdx.x] = 0;
#endif

	__syncwarp();

	int msg_sum = 0;
	for (uint32_t j = threadIdx.y; j < vn_degree; j += UNROLL_NODES) 
	{
		uint32_t vn = variable_nodes[j];
		uint32_t idx_row = vn & 0xffffu;
		uint32_t s = vn >> 16;
		uint32_t msg_offset = idx_row + idx_col * num_rows;
		uint32_t msg_idx = msg_offset * Zc + (i - s + (Zc << 8)) % Zc;
		msg_sum += llr_msg[msg_idx];
	}

	__syncwarp();
	if (threadIdx.y == 0) {
        if (idx_col < num_cols)
        msg_sum += llr_ch[idx_col*Zc + i];
    }


	msg_sum = min(max(msg_sum, -MAX_LLR_ACCUMULATOR_VALUE), MAX_LLR_ACCUMULATOR_VALUE);

#if UNROLL_NODES > 1
	msg_sums[threadIdx.y][threadIdx.x] = msg_sum;

	__syncthreads();

	if (threadIdx.y == 0) {
		msg_sum = 0;
		for (int k = 0; k < UNROLL_NODES; ++k)
			msg_sum += msg_sums[k][threadIdx.x];
		msg_sums[0][threadIdx.x] = msg_sum;
	}

	__syncthreads();

	msg_sum = msg_sums[0][threadIdx.x];
#endif

	//msg_sum = min(max(msg_sum, -MAX_LLR_ACCUMULATOR_VALUE), MAX_LLR_ACCUMULATOR_VALUE);

	__syncwarp();
	 if (idx_col < num_cols)
    llr_total[idx_col*Zc + i] = llr_accumulator_t(msg_sum);
}

// ---------------------------------------------------------------------------
// compute_syndrome_kernel
// ---------------------------------------------------------------------------
__launch_bounds__(512, 3)
static __global__ void compute_syndrome_kernel(
	llr_accumulator_t const* __restrict__ llr_total,
	uint32_t* __restrict__ syndrome,
	uint32_t Zc,
	uint32_t const* __restrict__ bg_cn,
	uint32_t const* __restrict__ bg_cn_degree,
	uint32_t cn_stride,
	uint32_t num_rows)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t i = tid % Zc;
	const uint32_t idx_row = tid / Zc;
	if (idx_row >= num_rows) return;

	const uint32_t cn_degree = bg_cn_degree[idx_row];
	uint32_t const* check_nodes = &bg_cn[idx_row * cn_stride];

	__syncwarp();

	uint32_t sign = 0;
	for (uint32_t ii = 0; ii < cn_degree; ++ii) {
		uint32_t cn = check_nodes[ii];
		uint32_t idx_col = cn & 0xffffu;
		uint32_t s = cn >> 16;
		sign ^= (uint32_t)(llr_total[idx_col * Zc + (i + s) % Zc] < 0);
	}

	sign = __any_sync(0xffffffff, sign);
	if (threadIdx.x % 32 == 0)
		syndrome[tid / 32] = sign;
}

// ---------------------------------------------------------------------------
// pack_bits_kernel
// ---------------------------------------------------------------------------
static constexpr uint32_t PACK_BITS_THREADS = 256;

static __global__ void hard_decision_kernel(
	llr_accumulator_t const*  llr_total,
	int32_t*  bits,
	uint32_t block_length)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < block_length)
		bits[tid] = (llr_total[tid] < 0) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Upload one table variant's four arrays into GPU memory and populate the
// corresponding pointer + stride fields of the provided base graph struct.
// Called once from ldpc_decoder_init().
// ---------------------------------------------------------------------------
static void upload_tables(Std_5G_base_graph& bg) {
	// Select the right host-side tables using bg.Bg and bg.index_list
	const uint32_t ils = bg.index_list;

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

	const uint32_t b = bg.Bg - 1; // 0-indexed

	uint32_t* tmp;

	CHECK_CUDA(cudaMalloc(&bg.cn_degree, sz_cn_degree[b][ils] * sizeof(uint32_t)));
	CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg.cn_degree), h_cn_degree[b][ils], sz_cn_degree[b][ils], cudaMemcpyHostToDevice));

	CHECK_CUDA(cudaMalloc(&bg.vn_degree, sz_vn_degree[b][ils] * sizeof(uint32_t)));
	CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg.vn_degree), h_vn_degree[b][ils], sz_vn_degree[b][ils], cudaMemcpyHostToDevice));

	CHECK_CUDA(cudaMalloc(&bg.cn, sz_cn[b][ils] * sizeof(uint32_t)));
	CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg.cn), h_cn[b][ils], sz_cn[b][ils], cudaMemcpyHostToDevice));
	bg.cn_stride = sz_cn[b][ils] / (sizeof(bg.cn[0]) * bg.num_rows);

	CHECK_CUDA(cudaMalloc(&bg.vn, sz_vn[b][ils] * sizeof(uint32_t)));
	CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg.vn), h_vn[b][ils], sz_vn[b][ils], cudaMemcpyHostToDevice));
	bg.vn_stride = sz_vn[b][ils] / (sizeof(bg.vn[0]) * bg.num_cols);
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init_context — per-thread working buffer allocation.
// Buffer sizes are derived from the already-resolved g_bg fields.
// ---------------------------------------------------------------------------
static ThreadContext* ldpc_decoder_init_context(int make_stream) 
{
	ThreadContext* ctx = new ThreadContext;

	const uint32_t num_vns = g_bg.num_cols * g_bg.Zc;
	const uint32_t num_cns = g_bg.num_rows * g_bg.Zc;

	if (make_stream) 
	{
		int hi = 0;
		cudaDeviceGetStreamPriorityRange(nullptr, &hi);
		CHECK_CUDA(cudaStreamCreateWithPriority(&ctx->stream, cudaStreamNonBlocking, hi));
	}

	CHECK_CUDA(cudaHostAlloc(&ctx->llr_in_buffer,
		num_vns * sizeof(int8_t),
		cudaHostAllocMapped | cudaHostAllocWriteCombined));

	CHECK_CUDA(cudaHostAlloc(&ctx->llr_bits_out_buffer,
		(g_bg.K_LDPC) * sizeof(int),
		cudaHostAllocMapped));

	CHECK_CUDA(cudaHostAlloc(&ctx->syndrome_buffer,
		(num_cns / 32) * sizeof(uint32_t),
		cudaHostAllocMapped));

	CHECK_CUDA(cudaMalloc(&ctx->llr_msg_buffer,
		g_bg.num_rows * g_bg.num_cols * g_bg.Zc * sizeof(llr_msg_t)));

	CHECK_CUDA(cudaMalloc(&ctx->llr_total_buffer,
		num_vns * sizeof(llr_accumulator_t)));
	return ctx;
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
ThreadContext* sp_cuda::ldpc_decoder_init(int K, int N, int make_stream) 
{
	// Delegate all BG selection and Zc resolution to your helper.
	// Throws std::invalid_argument on bad (K, N) — let it propagate.
	g_bg = build_5G_base_graph(K, N);

	printf("LDPC init: K=%d N=%d -> BG%u Zc=%u ils=%u rows=%u cols=%u\n",
		K, N, g_bg.Bg, g_bg.Zc, g_bg.index_list,
		g_bg.num_rows, g_bg.num_cols);

	// Upload only the one (Bg, index_list) table variant we need
	upload_tables(g_bg);

	g_initialized = true;
	return ldpc_decoder_init_context(make_stream);
}

// ---------------------------------------------------------------------------
// ldpc_decode
//
// Parameters simplified vs the original:
//   - BG and Z are gone (read from g_bg)
//   - block_length kept as K (the caller's original K, not K_LDPC)
// ---------------------------------------------------------------------------
uint32_t sp_cuda::ldpc_decode(
	ThreadContext* ctx,
	int8_t const* llr_in,            // g_bg.num_cols * g_bg.Zc bytes
	uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
	uint32_t       num_iter,
	uint32_t       perform_syndrome_check,
	int* llr_bits_out)
{
	cudaStream_t stream = ctx->stream;

	const uint32_t Zc = g_bg.Zc;
	const uint32_t num_vns = g_bg.num_cols * Zc;
	const uint32_t num_cns = g_bg.num_rows * Zc;

	memcpy(ctx->llr_in_buffer, llr_in, num_vns * sizeof(int8_t));

	const dim3 thread2d(NODE_KERNEL_BLOCK, UNROLL_NODES);
	int8_t const* llr_total = ctx->llr_in_buffer;

	for (uint32_t iter = 0; iter < num_iter; ++iter)
	{
		update_cn_kernel << <blocks_for(num_cns, NODE_KERNEL_BLOCK), thread2d, 0, stream >> > (
			llr_total, ctx->llr_msg_buffer,
			Zc,
			g_bg.cn, g_bg.cn_degree, g_bg.cn_stride,
			g_bg.num_rows,
			iter == 0);

		update_vn_kernel << <blocks_for(num_vns, NODE_KERNEL_BLOCK), thread2d, 0, stream >> > (
			ctx->llr_msg_buffer, ctx->llr_in_buffer, ctx->llr_total_buffer,
			Zc,
			g_bg.vn, g_bg.vn_degree, g_bg.vn_stride,
			g_bg.num_cols, g_bg.num_rows);

		llr_total = ctx->llr_total_buffer;
	}
	hard_decision_kernel << <blocks_for(K, PACK_BITS_THREADS), PACK_BITS_THREADS, 0, stream >> > (
		llr_total, ctx->llr_bits_out_buffer, K);

	cudaStreamSynchronize(stream);
	memcpy(llr_bits_out, ctx->llr_bits_out_buffer, K * sizeof(int));

	if (perform_syndrome_check) {
		compute_syndrome_kernel << <blocks_for(num_cns, 512), 512, 0, stream >> > (
			ctx->llr_total_buffer, ctx->syndrome_buffer,
			Zc,
			g_bg.cn, g_bg.cn_degree, g_bg.cn_stride,
			g_bg.num_rows);

		cudaStreamSynchronize(stream);

		for (uint32_t i = 0; i < num_cns / 32; ++i)
			if (ctx->syndrome_buffer[i] != 0)
				return num_iter + 1;
	}

	return num_iter - 1;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_shutdown
// ---------------------------------------------------------------------------
void sp_cuda::ldpc_decoder_shutdown() {
	cudaDeviceSynchronize();

	// Free the four GPU table allocations stored in g_bg
	cudaFree(const_cast<uint32_t*>(g_bg.cn_degree));
	cudaFree(const_cast<uint32_t*>(g_bg.vn_degree));
	cudaFree(const_cast<uint32_t*>(g_bg.cn));
	cudaFree(const_cast<uint32_t*>(g_bg.vn));

	g_initialized = false;
	g_bg = {};
}
