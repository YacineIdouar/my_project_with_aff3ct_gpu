/*
Decoder_LDPC_cuda_batched_kernel.cu
-----------------------------------
CUDA part of Decoder_LDPC_BP_flooding_cuda_batched: the kernels of Decoder_LDPC_cuda_kernel.cu with the frame as grid
dimension y. Each kernel offsets its pointers by 'frame * stride' and is otherwise the original: same arithmetic, same
operation order, so the same decoded bits.

Strides (in elements): the input LLRs and the output bits use the frame strides of the sockets, the work buffers
'num_cols * Zc' (totals) and 'num_rows * num_cols * Zc' (messages).
*/

#include <cstdint>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "Cuda/Decoder_LDPC_cuda_batched_kernel.hpp"
#include "Device/Cuda/Cuda_dispatch.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg1.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg2.hpp"

using aff3ct::tools::Std_5G_base_graph;

namespace
{
constexpr float MAX_CHANNEL_LLR = 128.0f;
constexpr float OMS_OFFSET = 0.5f;
constexpr uint32_t NODE_KERNEL_BLOCK = 128;
constexpr uint32_t UNROLL_NODES = 1;

void
check(const cudaError_t error, const char* what)
{
    if (error == cudaSuccess) return;
    std::stringstream message;
    message << what << ": " << cudaGetErrorString(error) << ".";
    throw spu::tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
}

inline uint32_t
blocks_for(const uint32_t n, const uint32_t bs)
{
    return (n + bs - 1) / bs;
}

__global__ void
clip_channel_kernel(const float* __restrict__ in,
                    const uint32_t in_stride,
                    float* __restrict__ out,
                    const uint32_t out_stride,
                    const uint32_t n)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const size_t f = blockIdx.y;
    out[f * out_stride + tid] = fmaxf(-MAX_CHANNEL_LLR, fminf(MAX_CHANNEL_LLR, in[f * in_stride + tid]));
}

// 'llr_total' is the channel LLRs on the first iteration (stride: the socket's), the totals afterwards
__global__ void
update_cn_kernel(const float* llr_total,
                 const uint32_t total_stride,
                 float* llr_msg,
                 const uint32_t msg_stride,
                 const uint32_t Zc,
                 const uint32_t* bg_cn,
                 const uint32_t* bg_cn_degree,
                 const uint32_t cn_stride,
                 const uint32_t num_rows,
                 const bool first_iter)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i = tid % Zc;
    const uint32_t idx_row = tid / Zc;
    if (idx_row >= num_rows) return;

    const size_t f = blockIdx.y;
    llr_total += f * total_stride;
    llr_msg += f * msg_stride;

    const uint32_t cn_degree = bg_cn_degree[idx_row];
    const uint32_t* check_nodes = &bg_cn[idx_row * cn_stride];

    float min_1 = 1e38f;
    float min_2 = 1e38f;
    int idx_min = -1;
    int node_sign = 1;
    uint32_t msg_signs = 0;

    for (uint32_t ii = threadIdx.y; ii < cn_degree; ii += UNROLL_NODES)
    {
        const uint32_t cn = check_nodes[ii];
        const uint32_t idx_col = cn & 0xffffu;
        const uint32_t s = cn >> 16;
        const uint32_t msg_offset = idx_row + idx_col * num_rows;
        const uint32_t msg_idx = msg_offset * Zc + i;

        float t = llr_total[idx_col * Zc + (i + s) % Zc];
        if (!first_iter) t -= llr_msg[msg_idx];

        const int sign = (t >= 0 ? 1 : -1);
        node_sign *= sign;
        msg_signs |= (t < 0) << ii;

        const float t_abs = fabsf(t);
        if (t_abs < min_1)
        {
            min_2 = min_1;
            min_1 = t_abs;
            idx_min = msg_idx;
        }
        else if (t_abs < min_2)
            min_2 = t_abs;
    }

    min_1 = fmaxf(0.f, min_1 - OMS_OFFSET);
    min_2 = fmaxf(0.f, min_2 - OMS_OFFSET);

    for (uint32_t ii = threadIdx.y; ii < cn_degree; ii += UNROLL_NODES)
    {
        const uint32_t cn = check_nodes[ii];
        const uint32_t idx_col = cn & 0xffffu;
        const uint32_t msg_offset = idx_row + idx_col * num_rows;
        const uint32_t msg_idx = msg_offset * Zc + i;
        const float min_val = (msg_idx == (uint32_t)idx_min) ? min_2 : min_1;
        const int msg_sign = (msg_signs >> ii) & 0x1 ? -1 : 1;
        llr_msg[msg_idx] = min_val * node_sign * msg_sign;
    }
}

__global__ void
update_vn_kernel(const float* llr_msg,
                 const uint32_t msg_stride,
                 const float* llr_ch,
                 const uint32_t ch_stride,
                 float* llr_total,
                 const uint32_t total_stride,
                 const uint32_t Zc,
                 const uint32_t* bg_vn,
                 const uint32_t* bg_vn_degree,
                 const uint32_t vn_stride,
                 const uint32_t num_cols,
                 const uint32_t num_rows)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i = tid % Zc;
    const uint32_t idx_col = tid / Zc;
    if (idx_col >= num_cols) return;

    const size_t f = blockIdx.y;
    llr_msg += f * msg_stride;
    llr_ch += f * ch_stride;
    llr_total += f * total_stride;

    const uint32_t vn_degree = bg_vn_degree[idx_col];
    const uint32_t* variable_nodes = &bg_vn[idx_col * vn_stride];

    float msg_sum = 0;
    for (uint32_t j = threadIdx.y; j < vn_degree; j += UNROLL_NODES)
    {
        const uint32_t vn = variable_nodes[j];
        const uint32_t idx_row = vn & 0xffffu;
        const uint32_t s = vn >> 16;
        const uint32_t msg_offset = idx_row + idx_col * num_rows;
        const uint32_t msg_idx = msg_offset * Zc + (i - s + (Zc << 8)) % Zc;
        msg_sum += llr_msg[msg_idx];
    }
    if (threadIdx.y == 0) msg_sum += llr_ch[idx_col * Zc + i];

    llr_total[idx_col * Zc + i] = msg_sum;
}

__global__ void
hard_decision_kernel(const float* llr_total,
                     const uint32_t total_stride,
                     int* bits,
                     const uint32_t bits_stride,
                     const uint32_t block_length)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= block_length) return;
    const size_t f = blockIdx.y;
    bits[f * bits_stride + tid] = (llr_total[f * total_stride + tid] < 0) ? 1 : 0;
}

// Uploads the four tables of the (Bg, index_list) variant and fills the pointer and stride fields of 'bg' (same
// selection as upload_tables() in Decoder_LDPC_cuda_kernel.cu).
void
upload_tables(Std_5G_base_graph& bg)
{
    const uint32_t ils = bg.index_list;
    const uint32_t* h_cn_degree[2][8] = { { BG1_CN_DEGREE_TABLE() }, { BG2_CN_DEGREE_TABLE() } };
    const uint32_t* h_vn_degree[2][8] = { { BG1_VN_DEGREE_TABLE() }, { BG2_VN_DEGREE_TABLE() } };
    const void* h_cn[2][8] = { { BG1_CN_TABLE() }, { BG2_CN_TABLE() } };
    const void* h_vn[2][8] = { { BG1_VN_TABLE() }, { BG2_VN_TABLE() } };
    const uint32_t sz_cn_degree[2][8] = { { BG1_CN_DEGREE_TABLE(sizeof) }, { BG2_CN_DEGREE_TABLE(sizeof) } };
    const uint32_t sz_vn_degree[2][8] = { { BG1_VN_DEGREE_TABLE(sizeof) }, { BG2_VN_DEGREE_TABLE(sizeof) } };
    const uint32_t sz_cn[2][8] = { { BG1_CN_TABLE(sizeof) }, { BG2_CN_TABLE(sizeof) } };
    const uint32_t sz_vn[2][8] = { { BG1_VN_TABLE(sizeof) }, { BG2_VN_TABLE(sizeof) } };
    const uint32_t b = bg.Bg - 1;

    auto upload = [](const uint32_t*& dst, const void* src, const size_t bytes)
    {
        void* ptr = nullptr;
        check(cudaMalloc(&ptr, bytes), "cudaMalloc (LDPC tables)");
        check(cudaMemcpy(ptr, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy (LDPC tables)");
        dst = static_cast<const uint32_t*>(ptr);
    };
    upload(bg.cn_degree, h_cn_degree[b][ils], sz_cn_degree[b][ils]);
    upload(bg.vn_degree, h_vn_degree[b][ils], sz_vn_degree[b][ils]);
    upload(bg.cn, h_cn[b][ils], sz_cn[b][ils]);
    upload(bg.vn, h_vn[b][ils], sz_vn[b][ils]);
    bg.cn_stride = sz_cn[b][ils] / (sizeof(uint32_t) * bg.num_rows);
    bg.vn_stride = sz_vn[b][ils] / (sizeof(uint32_t) * bg.num_cols);
}
}

sp_cuda::Cuda_decoder_batched::Cuda_decoder_batched(const int dev_id, const int K, const int N, const size_t n_frames)
  : dev_id(dev_id)
  , bg(aff3ct::tools::build_5G_base_graph(K, N))
{
    check(cudaSetDevice(this->dev_id), "cudaSetDevice");
    upload_tables(this->bg);
    this->reserve(n_frames);
}

sp_cuda::Cuda_decoder_batched::~Cuda_decoder_batched()
{
    cudaSetDevice(this->dev_id);
    cudaFree(this->llr_msg);
    cudaFree(this->llr_total);
    cudaFree(const_cast<uint32_t*>(this->bg.cn_degree));
    cudaFree(const_cast<uint32_t*>(this->bg.vn_degree));
    cudaFree(const_cast<uint32_t*>(this->bg.cn));
    cudaFree(const_cast<uint32_t*>(this->bg.vn));
}

void
sp_cuda::Cuda_decoder_batched::reserve(const size_t n_frames)
{
    if (n_frames == this->n_frames) return;
    check(cudaSetDevice(this->dev_id), "cudaSetDevice");
    cudaFree(this->llr_msg);
    cudaFree(this->llr_total);
    this->llr_msg = nullptr;
    this->llr_total = nullptr;

    const size_t num_vns = (size_t)this->bg.num_cols * this->bg.Zc;
    const size_t num_msgs = (size_t)this->bg.num_rows * this->bg.num_cols * this->bg.Zc;
    check(cudaMalloc(&this->llr_msg, n_frames * num_msgs * sizeof(float)), "cudaMalloc (LDPC messages)");
    check(cudaMalloc(&this->llr_total, n_frames * num_vns * sizeof(float)), "cudaMalloc (LDPC totals)");
    this->n_frames = n_frames;
}

std::vector<spu::device_interface::launch_timing>
sp_cuda::Cuda_decoder_batched::decode(const float* llr_in,
                                      const uint32_t in_stride,
                                      int* bits,
                                      const uint32_t bits_stride,
                                      const uint32_t n_frames,
                                      const uint32_t K,
                                      const uint32_t n_ite,
                                      spu::device_interface::GpuStream* stream)
{
    if (n_frames > this->n_frames) this->reserve(n_frames);

    const uint32_t Zc = this->bg.Zc;
    const uint32_t num_vns = this->bg.num_cols * Zc;
    const uint32_t num_cns = this->bg.num_rows * Zc;
    const uint32_t msg_stride = this->bg.num_rows * this->bg.num_cols * Zc;
    const dim3 thread2d(NODE_KERNEL_BLOCK, UNROLL_NODES);

    std::vector<spu::executor::CUDA_kernel_desc> chain;
    chain.reserve(2 + 2 * (size_t)n_ite);

    // kept as in the original chain: iteration 0 reads the raw channel LLRs, not the clipped ones
    chain.push_back(spu::sp_cuda::kernel(clip_channel_kernel,
                                         dim3(blocks_for(num_vns, 256), n_frames),
                                         256,
                                         0,
                                         llr_in,
                                         in_stride,
                                         this->llr_total,
                                         num_vns,
                                         num_vns));

    const float* llr_total = llr_in;
    uint32_t total_stride = in_stride;
    for (uint32_t iter = 0; iter < n_ite; ++iter)
    {
        chain.push_back(spu::sp_cuda::kernel(update_cn_kernel,
                                             dim3(blocks_for(num_cns, NODE_KERNEL_BLOCK), n_frames),
                                             thread2d,
                                             0,
                                             llr_total,
                                             total_stride,
                                             this->llr_msg,
                                             msg_stride,
                                             Zc,
                                             this->bg.cn,
                                             this->bg.cn_degree,
                                             this->bg.cn_stride,
                                             this->bg.num_rows,
                                             iter == 0));
        chain.push_back(spu::sp_cuda::kernel(update_vn_kernel,
                                             dim3(blocks_for(num_vns, NODE_KERNEL_BLOCK), n_frames),
                                             thread2d,
                                             0,
                                             this->llr_msg,
                                             msg_stride,
                                             llr_in,
                                             in_stride,
                                             this->llr_total,
                                             num_vns,
                                             Zc,
                                             this->bg.vn,
                                             this->bg.vn_degree,
                                             this->bg.vn_stride,
                                             this->bg.num_cols,
                                             this->bg.num_rows));
        llr_total = this->llr_total;
        total_stride = num_vns;
    }

    chain.push_back(spu::sp_cuda::kernel(hard_decision_kernel,
                                         dim3(blocks_for(K, 256), n_frames),
                                         256,
                                         0,
                                         llr_total,
                                         total_stride,
                                         bits,
                                         bits_stride,
                                         K));

    return spu::sp_cuda::dispatch(stream, chain);
}
