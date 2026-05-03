/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

HIP port of the LDPC decoder.
Changes vs the CUDA original:
  - All #include <cuda_runtime.h> / CUDA headers replaced with <hip/hip_runtime.h>
  - cudaXxx API calls replaced with hipXxx equivalents
  - __ldg() removed (HIP has no equivalent; plain dereference is fine on GCN/RDNA)
  - __any_sync() replaced with __any() — HIP's warp vote (no mask argument)
  - __popc() replaced with __builtin_popcount() (available on both clang and GCC host,
    and HIP device code exposes it as a builtin as well)
  - __shfl_xor_sync() removed together with the pack_bits_kernel (already commented
    out in the original); the hard_decision_kernel has no shuffle use
  - __syncwarp() removed — HIP has no lane-masking primitive and convergent warp
    execution makes it a no-op in practice; the surrounding logic is unchanged
  - __atomic_exchange with GCC __ATOMIC_* replaced with C++ std::atomic / explicit
    fences so the file compiles cleanly with both hipcc and g++
  - CHECK_CUDA macro renamed CHECK_HIP
  - Namespace sp_cuda renamed sp_hip
*/

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <climits>
#include <atomic>
#include <iostream>

#include "Decoder_LDPC_hip_kernel.hpp"
#include "ldpc_tables_bg1.hpp"
#include "ldpc_tables_bg2.hpp"
#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"

using namespace aff3ct::tools;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------
static constexpr int MAX_LLR_ACCUMULATOR_VALUE = 127;
typedef int8_t llr_accumulator_t;
static constexpr int MAX_LLR_MSG_VALUE = 127;
typedef int8_t llr_msg_t;

#define APPLY_DAMPING_INT(x) ((x) * 3 / 4)

// ---------------------------------------------------------------------------
// Single global base graph instance
// ---------------------------------------------------------------------------
static Std_5G_base_graph g_bg          = {};
static bool              g_initialized = false;

// ---------------------------------------------------------------------------
// Thread context (one per OS thread)
// ---------------------------------------------------------------------------
struct ThreadContext_hip {
    hipStream_t        stream              = nullptr;
    int8_t*            llr_in_buffer       = nullptr;
    int*               llr_bits_out_buffer = nullptr;
    uint32_t*          syndrome_buffer     = nullptr;
    llr_msg_t*         llr_msg_buffer      = nullptr;
    llr_accumulator_t* llr_total_buffer    = nullptr;
    ThreadContext_hip*     next                = nullptr;
};
static __thread ThreadContext_hip thread_ctx  = {};
// Linked list of all allocated contexts — protected by a plain pointer-swap
// using C++ std::atomic for portability.
static std::atomic<ThreadContext_hip*> all_contexts{nullptr};

// ---------------------------------------------------------------------------
// Error check helper
// ---------------------------------------------------------------------------
#define CHECK_HIP(call)                                                        \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess)                                                  \
            printf("HIP error %d: %s at %s:%d\n",                             \
                   static_cast<int>(_e), hipGetErrorString(_e),                \
                   __FILE__, __LINE__);                                         \
    } while (0)

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
//
// Removed vs CUDA original:
//   - __ldg()    → plain dereference (GCN L2 cache makes it equivalent)
//   - __syncwarp() → removed (no-op on convergent warps; HIP has no masked form)
//   - __popc()   → __builtin_popcount() (available in HIP device code via clang)
// ---------------------------------------------------------------------------
__launch_bounds__(UNROLL_NODES * NODE_KERNEL_BLOCK, 3)
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

    const uint32_t cn_degree    = bg_cn_degree[idx_row];
    uint32_t const* check_nodes = &bg_cn[idx_row * cn_stride];

    int      min_1     = INT_MAX;
    int      min_2     = INT_MAX;
    int      idx_min   = -1;
    uint32_t msg_signs = 0;

    // First pass: compute min-sum statistics
    for (uint32_t ii = 0; ii < cn_degree; ++ii) {
        uint32_t cn         = check_nodes[ii];
        uint32_t idx_col    = cn & 0xffffu;
        uint32_t s          = cn >> 16;
        uint32_t msg_offset = idx_row + idx_col * num_rows;
        uint32_t msg_idx    = msg_offset * Zc + i;

        // Plain load — no __ldg needed; hardware cache handles it.
        int t = llr_total[idx_col * Zc + (i + s) % Zc];
        if (!first_iter)
            t -= llr_msg[msg_idx];   // was: __ldg(&llr_msg[msg_idx])

        msg_signs |= (t < 0) << ii;

        int t_abs = abs(t);
        if (t_abs < min_1) {
            min_2   = min_1;
            min_1   = t_abs;
            idx_min = static_cast<int>(msg_idx);
        } else if (t_abs < min_2) {
            min_2 = t_abs;
        }
    }

    // Overall parity of the row
    // __popc → __builtin_popcount: same semantics, portable on HIP/clang.
    const int node_sign = (__builtin_popcount(msg_signs) & 1) ? -1 : 1;

    min_1 = min(max(APPLY_DAMPING_INT(min_1), -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);
    min_2 = min(max(APPLY_DAMPING_INT(min_2), -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);

    // Second pass: write messages
	for (uint32_t ii = threadIdx.y; ii < cn_degree; ++ii) {
        uint32_t cn         = check_nodes[ii];
        uint32_t idx_col    = cn & 0xffffu;
        uint32_t msg_offset = idx_row + idx_col * num_rows;
        uint32_t msg_idx    = msg_offset * Zc + i;

        int min_val  = (static_cast<int>(msg_idx) == idx_min) ? min_2 : min_1;
        int msg_sign = ((msg_signs >> ii) & 1) ? -1 : 1;

        llr_msg[msg_idx] = llr_msg_t(min_val * node_sign * msg_sign);
    }
}

// ---------------------------------------------------------------------------
// update_vn_kernel
//
// Removed vs CUDA original:
//   - __syncwarp() → removed (convergent warps; no masked form in HIP)
// ---------------------------------------------------------------------------
__launch_bounds__(UNROLL_NODES * NODE_KERNEL_BLOCK, 3)
static __global__ void update_vn_kernel(
        llr_msg_t const*         __restrict__ llr_msg,
        int8_t const*            __restrict__ llr_ch,
        llr_accumulator_t*       __restrict__ llr_total,
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

    const uint32_t vn_degree       = bg_vn_degree[idx_col];
    uint32_t const* variable_nodes = &bg_vn[idx_col * vn_stride];

    int msg_sum = 0;
    for (uint32_t j = threadIdx.y; j < vn_degree; ++j) {
        uint32_t vn         = variable_nodes[j];
        uint32_t idx_row    = vn & 0xffffu;
        uint32_t s          = vn >> 16;
        uint32_t msg_offset = idx_row + idx_col * num_rows;
        uint32_t msg_idx    = msg_offset * Zc + (i - s + (Zc << 8)) % Zc;
        msg_sum += llr_msg[msg_idx];
    }

   if (threadIdx.y == 0)
        msg_sum += llr_ch[idx_col * Zc + i];
    msg_sum  = min(max(msg_sum, -MAX_LLR_ACCUMULATOR_VALUE), MAX_LLR_ACCUMULATOR_VALUE);

    llr_total[idx_col * Zc + i] = llr_accumulator_t(msg_sum);
}

// ---------------------------------------------------------------------------
// hard_decision_kernel  (pack_bits_kernel was already commented out)
// No CUDA intrinsics were used here — unchanged.
// ---------------------------------------------------------------------------
static constexpr uint32_t PACK_BITS_THREADS = 256;

__launch_bounds__(PACK_BITS_THREADS, 6)
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
// upload_tables — identical logic, HIP malloc/memcpy
// ---------------------------------------------------------------------------
static void upload_tables(Std_5G_base_graph& bg) {
    const uint32_t ils = bg.index_list;

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

    const uint32_t b = bg.Bg - 1; // 0-indexed

    uint32_t* tmp;

    CHECK_HIP(hipMalloc(&tmp, sz_cn_degree[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_cn_degree[b][ils], sz_cn_degree[b][ils], hipMemcpyHostToDevice));
    bg.cn_degree = tmp;

    CHECK_HIP(hipMalloc(&tmp, sz_vn_degree[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_vn_degree[b][ils], sz_vn_degree[b][ils], hipMemcpyHostToDevice));
    bg.vn_degree = tmp;

    CHECK_HIP(hipMalloc(&tmp, sz_cn[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_cn[b][ils], sz_cn[b][ils], hipMemcpyHostToDevice));
    bg.cn        = tmp;
    bg.cn_stride = sz_cn[b][ils] / (sizeof(uint32_t) * bg.num_rows);

    CHECK_HIP(hipMalloc(&tmp, sz_vn[b][ils]));
    CHECK_HIP(hipMemcpy(tmp, h_vn[b][ils], sz_vn[b][ils], hipMemcpyHostToDevice));
    bg.vn        = tmp;
    bg.vn_stride = sz_vn[b][ils] / (sizeof(uint32_t) * bg.num_cols);
}

// ---------------------------------------------------------------------------
// ldpc_decoder_init_context — per-thread working buffer allocation
//
// Replaced GCC __atomic_exchange with std::atomic::exchange for portability.
// hipHostMalloc flags: hipHostMallocMapped | hipHostMallocWriteCombined
//   (maps to cudaHostAllocMapped | cudaHostAllocWriteCombined)
// ---------------------------------------------------------------------------
static ThreadContext_hip& ldpc_decoder_init_context(int make_stream) {
    auto& ctx = thread_ctx;
    if (ctx.llr_in_buffer) return ctx;

    const uint32_t num_vns = g_bg.num_cols * g_bg.Zc;
    const uint32_t num_cns = g_bg.num_rows * g_bg.Zc;

    if (make_stream) {
        int hi = 0;
        hipDeviceGetStreamPriorityRange(nullptr, &hi);
        CHECK_HIP(hipStreamCreateWithPriority(&ctx.stream, hipStreamNonBlocking, hi));
    }

    CHECK_HIP(hipHostMalloc(&ctx.llr_in_buffer,
              num_vns * sizeof(int8_t),
              hipHostMallocMapped | hipHostMallocWriteCombined));

    CHECK_HIP(hipHostMalloc(&ctx.llr_in_buffer,
              g_bg.K_LDPC * sizeof(int),
              hipHostMallocMapped));

    // syndrome buffer: one uint32 per warpSize lanes, across num_cns threads.
    // Allocate for the maximum wavefront width (64) to be safe; only
    // (num_cns / warpSize) entries are actually written at runtime.
    CHECK_HIP(hipHostMalloc(&ctx.syndrome_buffer,
              (num_cns / 32) * sizeof(uint32_t),   // 32 = minimum warpSize
              hipHostMallocMapped));

    CHECK_HIP(hipMalloc(&ctx.llr_msg_buffer,
              g_bg.num_rows * g_bg.num_cols * g_bg.Zc * sizeof(llr_msg_t)));

    CHECK_HIP(hipMalloc(&ctx.llr_total_buffer,
              num_vns * sizeof(llr_accumulator_t)));

    ThreadContext_hip* self = &ctx;
   	self->next = all_contexts.exchange(self, std::memory_order_acq_rel);
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

    // UNROLL_NODES == 1 → plain 1-D launch (no 2-D thread block needed)
    int8_t const* llr_total = ctx.llr_in_buffer;

    for (uint32_t iter = 0; iter < num_iter; ++iter) {
        hipLaunchKernelGGL(update_cn_kernel,
            dim3(blocks_for(num_cns, NODE_KERNEL_BLOCK)), dim3(NODE_KERNEL_BLOCK),
            0, stream,
            llr_total, ctx.llr_msg_buffer,
            Zc,
            g_bg.cn, g_bg.cn_degree, g_bg.cn_stride,
            g_bg.num_rows,
            iter == 0);

        hipLaunchKernelGGL(update_vn_kernel,
            dim3(blocks_for(num_vns, NODE_KERNEL_BLOCK)), dim3(NODE_KERNEL_BLOCK),
            0, stream,
            ctx.llr_msg_buffer, ctx.llr_in_buffer, ctx.llr_total_buffer,
            Zc,
            g_bg.vn, g_bg.vn_degree, g_bg.vn_stride,
            g_bg.num_cols, g_bg.num_rows);

        llr_total = ctx.llr_total_buffer;
    }

    hipLaunchKernelGGL(hard_decision_kernel,
        dim3(blocks_for(K, PACK_BITS_THREADS)), dim3(PACK_BITS_THREADS),
        0, stream,
        llr_total, ctx.llr_bits_out_buffer, K);

    CHECK_HIP(hipStreamSynchronize(stream));
    memcpy(llr_bits_out, ctx.llr_bits_out_buffer, K * sizeof(int));

    return num_iter - 1;
}

// ---------------------------------------------------------------------------
// ldpc_decoder_shutdown
// ---------------------------------------------------------------------------
void sp_hip::ldpc_decoder_shutdown() {
    CHECK_HIP(hipDeviceSynchronize());

    // Drain the linked list
    ThreadContext_hip* ctx = all_contexts.exchange(nullptr, std::memory_order_acq_rel);
    while (ctx) {
        ThreadContext_hip* next = ctx->next;
        hipHostFree(ctx->llr_in_buffer);
        hipHostFree(ctx->llr_bits_out_buffer);
        hipHostFree(ctx->syndrome_buffer);
        hipFree(ctx->llr_msg_buffer);
        hipFree(ctx->llr_total_buffer);
        if (ctx->stream) hipStreamDestroy(ctx->stream);
        ctx = next;
    }

    // Free the four GPU table allocations stored in g_bg
    hipFree(const_cast<uint32_t*>(g_bg.cn_degree));
    hipFree(const_cast<uint32_t*>(g_bg.vn_degree));
    hipFree(const_cast<uint32_t*>(g_bg.cn));
    hipFree(const_cast<uint32_t*>(g_bg.vn));

    g_initialized = false;
    g_bg = {};
}
