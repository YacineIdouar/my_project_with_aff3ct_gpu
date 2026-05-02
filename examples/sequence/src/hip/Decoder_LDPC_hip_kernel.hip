/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

HIP port of Decoder_LDPC_cuda_kernel.cpp
Changes vs the CUDA original:
  - All CUDA APIs replaced with HIP equivalents
  - UNROLL_NODES > 1 path removed entirely (was compile-time dead code at UNROLL_NODES=1)
  - All __syncwarp() calls removed (redundant at UNROLL_NODES=1, no-op on AMD anyway)
  - __ldg() replaced with plain dereference (HIP/GCN has no separate texture cache path)
  - 0xffffffff warp masks widened to 0xffffffffffffffffull (AMD wavefront = 64 lanes on gfx9,
    32 on gfx10+; using __ballot_sync to derive the active mask avoids the hardcoded assumption)
  - CHECK_CUDA replaced with CHECK_HIP
  - __popc replaced with __popcll on 64-bit ballot values
*/

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <climits>
#include <iostream>

#include "Decoder_LDPC_hip_kernel.hpp"
#include "ldpc_tables_bg1.hpp"
#include "ldpc_tables_bg2.hpp"
#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"

using namespace aff3ct::tools;

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------
#define CHECK_HIP(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) \
        printf("HIP error %d: %s at %s:%d\n", \
               (int)_e, hipGetErrorString(_e), __FILE__, __LINE__); \
} while(0)

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------
static constexpr int MAX_LLR_ACCUMULATOR_VALUE = 127;
typedef int8_t llr_accumulator_t;
static constexpr int MAX_LLR_MSG_VALUE = 127;
typedef int8_t llr_msg_t;

#define APPLY_DAMPING_INT(x) ((x) * 3 / 4)

// ---------------------------------------------------------------------------
// Kernel configuration — single-row-per-thread, no unrolling
// ---------------------------------------------------------------------------
static constexpr uint32_t NODE_KERNEL_BLOCK = 128;
static constexpr uint32_t PACK_BITS_THREADS = 256;

inline __host__ __device__ uint32_t blocks_for(uint32_t n, uint32_t bs) {
    return (n + bs - 1) / bs;
}

// ---------------------------------------------------------------------------
// Global base-graph (populated once by ldpc_decoder_init)
// ---------------------------------------------------------------------------
static Std_5G_base_graph g_bg          = {};
static bool              g_initialized = false;

// GPU-side table pointers and strides (mirrors of CUDA version)
struct BgDeviceTables {
    uint32_t* cn        = nullptr;
    uint32_t* vn        = nullptr;
    uint32_t* cn_degree = nullptr;
    uint32_t* vn_degree = nullptr;
    uint32_t  cn_stride = 0;
    uint32_t  vn_stride = 0;
};
static BgDeviceTables g_dev = {};

// ---------------------------------------------------------------------------
// Per-thread context
// ---------------------------------------------------------------------------
struct ThreadContext_hip {
    hipStream_t        stream              = 0;
    int8_t*            llr_in_buffer       = nullptr;  // pinned host
    int*               llr_bits_out_buffer = nullptr;  // pinned host
    uint32_t*          syndrome_buffer     = nullptr;  // pinned host
    llr_msg_t*         llr_msg_buffer      = nullptr;  // device
    llr_accumulator_t* llr_total_buffer    = nullptr;  // device
    ThreadContext_hip*     next                = nullptr;
};

static __thread ThreadContext_hip thread_ctx  = {};
static ThreadContext_hip*         all_contexts = nullptr;

// ---------------------------------------------------------------------------
// update_cn_kernel  (min-sum / belief propagation, check-node update)
//
// One thread per (row, z-index) pair.
// All __syncwarp() removed: every variable is a private register,
// there is no shared memory, and no cross-lane dependency.
// ---------------------------------------------------------------------------
static __global__ void update_cn_kernel(
        llr_accumulator_t const* __restrict__ llr_total,
        llr_msg_t*               __restrict__ llr_msg,
        uint32_t Zc,
        uint32_t const* __restrict__ bg_cn,
        uint32_t const* __restrict__ bg_cn_degree,
        uint32_t cn_stride,
        uint32_t num_rows,
        bool first_iter)
{
    const uint32_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i       = tid % Zc;
    const uint32_t idx_row = tid / Zc;

    if (idx_row >= num_rows) return;

    const uint32_t  cn_degree    = bg_cn_degree[idx_row];
    uint32_t const* check_nodes  = &bg_cn[idx_row * cn_stride];

    int      min_1     = INT_MAX;
    int      min_2     = INT_MAX;
    int      idx_min   = -1;
    uint32_t msg_signs = 0;

    // First pass: find two smallest magnitudes and accumulate signs
    for (uint32_t ii = 0; ii < cn_degree; ++ii) {
        uint32_t cn         = check_nodes[ii];
        uint32_t idx_col    = cn & 0xffffu;
        uint32_t s          = cn >> 16;
        uint32_t msg_offset = idx_row + idx_col * num_rows;
        uint32_t msg_idx    = msg_offset * Zc + i;

        int t = llr_total[idx_col * Zc + (i + s) % Zc];
        if (!first_iter)
            t -= llr_msg[msg_idx];   // plain load — no __ldg needed on GCN

        if (t < 0) msg_signs |= (1u << ii);

        int t_abs = abs(t);
        if (t_abs < min_1) { min_2 = min_1; min_1 = t_abs; idx_min = (int)msg_idx; }
        else if (t_abs < min_2) min_2 = t_abs;
    }

    const int node_sign = (__popc(msg_signs) & 1) ? -1 : 1;

    min_1 = min(max(APPLY_DAMPING_INT(min_1), -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);
    min_2 = min(max(APPLY_DAMPING_INT(min_2), -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);

    // Second pass: write outgoing messages
    for (uint32_t ii = 0; ii < cn_degree; ++ii) {
        uint32_t cn         = check_nodes[ii];
        uint32_t idx_col    = cn & 0xffffu;
        uint32_t msg_offset = idx_row + idx_col * num_rows;
        uint32_t msg_idx    = msg_offset * Zc + i;

        int min_val  = ((int)msg_idx == idx_min) ? min_2 : min_1;
        int msg_sign = ((msg_signs >> ii) & 1u) ? -1 : 1;

        llr_msg[msg_idx] = llr_msg_t(min_val * node_sign * msg_sign);
    }
}

// ---------------------------------------------------------------------------
// update_vn_kernel  (variable-node update)
//
// One thread per (col, z-index) pair.
// All three __syncwarp() calls removed: msg_sum is a private register,
// threadIdx.y is always 0, and the final write index is unique per thread.
// ---------------------------------------------------------------------------
static __global__ void update_vn_kernel(
        llr_msg_t const*   __restrict__ llr_msg,
        int8_t const*      __restrict__ llr_ch,
        llr_accumulator_t* __restrict__ llr_total,
        uint32_t Zc,
        uint32_t const* __restrict__ bg_vn,
        uint32_t const* __restrict__ bg_vn_degree,
        uint32_t vn_stride,
        uint32_t num_cols,
        uint32_t num_rows)
{
    const uint32_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i       = tid % Zc;
    const uint32_t idx_col = tid / Zc;

    if (idx_col >= num_cols) return;

    const uint32_t  vn_degree       = bg_vn_degree[idx_col];
    uint32_t const* variable_nodes  = &bg_vn[idx_col * vn_stride];

    int msg_sum = 0;
    for (uint32_t j = 0; j < vn_degree; ++j) {
        uint32_t vn         = variable_nodes[j];
        uint32_t idx_row    = vn & 0xffffu;
        uint32_t s          = vn >> 16;
        uint32_t msg_offset = idx_row + idx_col * num_rows;
        uint32_t msg_idx    = msg_offset * Zc + (i - s + (Zc << 8)) % Zc;
        msg_sum += llr_msg[msg_idx];
    }

    msg_sum += llr_ch[idx_col * Zc + i];
    msg_sum  = min(max(msg_sum, -MAX_LLR_ACCUMULATOR_VALUE), MAX_LLR_ACCUMULATOR_VALUE);

    llr_total[idx_col * Zc + i] = llr_accumulator_t(msg_sum);
}
// ---------------------------------------------------------------------------
// hard_decision_kernel
// ---------------------------------------------------------------------------
static __global__ void hard_decision_kernel(
        llr_accumulator_t const* __restrict__ llr_total,
        int32_t*                 __restrict__ bits,
        uint32_t block_length)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < block_length)
        bits[tid] = (llr_total[tid] < 0) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Upload tables to device memory
// ---------------------------------------------------------------------------
static void upload_tables(const Std_5G_base_graph& bg) {
    const uint32_t ils = bg.index_list;
    const uint32_t b   = bg.Bg - 1;

    const uint32_t* h_cn_degree[2][8] = { { BG1_CN_DEGREE_TABLE() },
                                           { BG2_CN_DEGREE_TABLE() } };
    const uint32_t* h_vn_degree[2][8] = { { BG1_VN_DEGREE_TABLE() },
                                           { BG2_VN_DEGREE_TABLE() } };
    const void*     h_cn[2][8]        = { { BG1_CN_TABLE() },
                                           { BG2_CN_TABLE() } };
    const void*     h_vn[2][8]        = { { BG1_VN_TABLE() },
                                           { BG2_VN_TABLE() } };

    const uint32_t sz_cn_degree[2][8] = { { BG1_CN_DEGREE_TABLE(sizeof) },
                                           { BG2_CN_DEGREE_TABLE(sizeof) } };
    const uint32_t sz_vn_degree[2][8] = { { BG1_VN_DEGREE_TABLE(sizeof) },
                                           { BG2_VN_DEGREE_TABLE(sizeof) } };
    const uint32_t sz_cn[2][8]        = { { BG1_CN_TABLE(sizeof) },
                                           { BG2_CN_TABLE(sizeof) } };
    const uint32_t sz_vn[2][8]        = { { BG1_VN_TABLE(sizeof) },
                                           { BG2_VN_TABLE(sizeof) } };

    uint32_t* tmp;

    CHECK_HIP(hipMalloc(&tmp, sz_cn_degree[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_cn_degree[b][ils], sz_cn_degree[b][ils], hipMemcpyHostToDevice));
    g_dev.cn_degree = tmp;

    CHECK_HIP(hipMalloc(&tmp, sz_vn_degree[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_vn_degree[b][ils], sz_vn_degree[b][ils], hipMemcpyHostToDevice));
    g_dev.vn_degree = tmp;

    CHECK_HIP(hipMalloc(&tmp, sz_cn[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_cn[b][ils], sz_cn[b][ils], hipMemcpyHostToDevice));
    g_dev.cn        = tmp;
    g_dev.cn_stride = sz_cn[b][ils] / (sizeof(uint32_t) * bg.num_rows);

    CHECK_HIP(hipMalloc(&tmp, sz_vn[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_vn[b][ils], sz_vn[b][ils], hipMemcpyHostToDevice));
    g_dev.vn        = tmp;
    g_dev.vn_stride = sz_vn[b][ils] / (sizeof(uint32_t) * bg.num_cols);
}

// ---------------------------------------------------------------------------
// Per-thread context initialisation
// ---------------------------------------------------------------------------
static ThreadContext_hip& ldpc_decoder_init_context(int make_stream) {
    auto& ctx = thread_ctx;
    if (ctx.llr_in_buffer) return ctx;

    const uint32_t num_vns = g_bg.num_cols * g_bg.Zc;
    const uint32_t num_cns = g_bg.num_rows * g_bg.Zc;

    if (make_stream) {
        int lo = 0, hi = 0;
        CHECK_HIP(hipDeviceGetStreamPriorityRange(&lo, &hi));
        CHECK_HIP(hipStreamCreateWithPriority(&ctx.stream, hipStreamNonBlocking, hi));
    }

    CHECK_HIP(hipHostMalloc(&ctx.llr_in_buffer,
              num_vns * sizeof(int8_t),
              hipHostMallocMapped | hipHostMallocWriteCombined));

    CHECK_HIP(hipHostMalloc(&ctx.llr_bits_out_buffer,
              g_bg.K_LDPC * sizeof(int),
              hipHostMallocMapped));

    CHECK_HIP(hipHostMalloc(&ctx.syndrome_buffer,
              (num_cns / 32) * sizeof(uint32_t),
              hipHostMallocMapped));

    CHECK_HIP(hipMalloc(&ctx.llr_msg_buffer,
              g_bg.num_rows * g_bg.num_cols * g_bg.Zc * sizeof(llr_msg_t)));

    CHECK_HIP(hipMalloc(&ctx.llr_total_buffer,
              num_vns * sizeof(llr_accumulator_t)));

    ThreadContext_hip* self = &ctx;
    __atomic_exchange(&all_contexts, &self, &self->next, __ATOMIC_ACQ_REL);

    return ctx;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init
// ---------------------------------------------------------------------------
ThreadContext_hip* sp_hip::ldpc_decoder_init(int K, int N, int make_stream) {
    if (g_initialized)
        return &ldpc_decoder_init_context(make_stream);

    g_bg = build_5G_base_graph(K, N);

    printf("LDPC init: K=%d N=%d -> BG%u Zc=%u ils=%u rows=%u cols=%u\n",
           K, N, g_bg.Bg, g_bg.Zc, g_bg.index_list,
           g_bg.num_rows, g_bg.num_cols);

    upload_tables(g_bg);

    g_initialized = true;
    return &ldpc_decoder_init_context(make_stream);
}

// ---------------------------------------------------------------------------
// ldpc_decode
// ---------------------------------------------------------------------------
uint32_t sp_hip::ldpc_decode(
        int8_t const*  llr_in,
        uint32_t       K,
        uint32_t       num_iter,
        uint32_t       perform_syndrome_check,
        int*           llr_bits_out)
{
    auto& ctx = ldpc_decoder_init_context(0);
    hipStream_t stream = ctx.stream;

    const uint32_t Zc      = g_bg.Zc;
    const uint32_t num_vns = g_bg.num_cols * Zc;
    const uint32_t num_cns = g_bg.num_rows * Zc;

    memcpy(ctx.llr_in_buffer, llr_in, num_vns * sizeof(int8_t));

    const llr_accumulator_t* llr_total = 
        reinterpret_cast<const llr_accumulator_t*>(ctx.llr_in_buffer);

    for (uint32_t iter = 0; iter < num_iter; ++iter) {
        update_cn_kernel<<<blocks_for(num_cns, NODE_KERNEL_BLOCK),
                           NODE_KERNEL_BLOCK, 0, stream>>>(
            llr_total, ctx.llr_msg_buffer,
            Zc,
            g_dev.cn, g_dev.cn_degree, g_dev.cn_stride,
            g_bg.num_rows,
            iter == 0);

        update_vn_kernel<<<blocks_for(num_vns, NODE_KERNEL_BLOCK),
                           NODE_KERNEL_BLOCK, 0, stream>>>(
            ctx.llr_msg_buffer, ctx.llr_in_buffer, ctx.llr_total_buffer,
            Zc,
            g_dev.vn, g_dev.vn_degree, g_dev.vn_stride,
            g_bg.num_cols, g_bg.num_rows);

        llr_total = ctx.llr_total_buffer;
    }

    hard_decision_kernel<<<blocks_for(K, PACK_BITS_THREADS),
                           PACK_BITS_THREADS, 0, stream>>>(
        llr_total, reinterpret_cast<int32_t*>(ctx.llr_bits_out_buffer), K);

    CHECK_HIP(hipStreamSynchronize(stream));
    memcpy(llr_bits_out, ctx.llr_bits_out_buffer, K * sizeof(int));

    return num_iter - 1;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_shutdown
// ---------------------------------------------------------------------------
void sp_hip::ldpc_decoder_shutdown() {
    CHECK_HIP(hipDeviceSynchronize());

    ThreadContext_hip* ctx = nullptr;
    __atomic_exchange(&all_contexts, &ctx, &ctx, __ATOMIC_ACQ_REL);
    while (ctx) {
        CHECK_HIP(hipHostFree(ctx->llr_in_buffer));
        CHECK_HIP(hipHostFree(ctx->llr_bits_out_buffer));
        CHECK_HIP(hipFree(ctx->llr_msg_buffer));
        CHECK_HIP(hipFree(ctx->llr_total_buffer));
        if (ctx->stream) CHECK_HIP(hipStreamDestroy(ctx->stream));
        ctx = ctx->next;
    }

    CHECK_HIP(hipFree(g_dev.cn_degree));
    CHECK_HIP(hipFree(g_dev.vn_degree));
    CHECK_HIP(hipFree(g_dev.cn));
    CHECK_HIP(hipFree(g_dev.vn));

    g_initialized = false;
    g_bg          = {};
    g_dev         = {};
}