/*
Decoder_LDPC_vulkan_batched_kernel.cpp
--------------------------------------
Vulkan part of Decoder_LDPC_BP_flooding_vulkan_batched: the dispatch chain of Decoder_LDPC_vulkan_kernel.cpp with the
frame as work group y (Shaders/ldpc_*_batched.comp), as Decoder_LDPC_cuda_batched_kernel.cu does with grid y on CUDA.
Each shader offsets its indices by 'frame * stride' and is otherwise the per-frame one: same arithmetic, same
operation order, so the same decoded bits.

Strides (in elements): the input LLRs and the output bits use the frame strides of the sockets, the work buffers
'num_cols * Zc' (totals) and 'num_rows * num_cols * Zc' (messages).

Vulkan binds a whole buffer to a descriptor, so the work buffers of n_f frames must fit in the device's
maxStorageBufferRange (4 GiB on desktop drivers, 128 MiB on some others): BG1 with Zc = 384 takes 4.8 MB of messages
per frame.
*/

#ifdef DECODER_VULKAN

#include <cstdint>
#include <tuple>
#include <vector>
#include <vulkan/vulkan.h>

#include "Vulkan/Decoder_LDPC_vulkan_batched_kernel.hpp"
#include "Device/Devices_manager.hpp"
#include "Device/Vulkan/Vulkan_device.hpp"
#include "Device/Vulkan/Vulkan_dispatch.hpp"
#include "Device/Vulkan/Vulkan_executor.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg1.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg2.hpp"

#include "ldpc_clip_channel_batched_spv.hpp"  // generated at build time from Shaders/ldpc_clip_channel_batched.comp
#include "ldpc_update_cn_batched_spv.hpp"     // generated at build time from Shaders/ldpc_update_cn_batched.comp
#include "ldpc_update_vn_batched_spv.hpp"     // generated at build time from Shaders/ldpc_update_vn_batched.comp
#include "ldpc_hard_decision_batched_spv.hpp" // generated at build time from Shaders/ldpc_hard_decision_batched.comp

using aff3ct::tools::Std_5G_base_graph;
using spu::device_interface::compute_api;
using spu::device_interface::memory_type;

namespace
{
// local_size_x of the shaders (NODE_KERNEL_BLOCK of the CUDA kernels for the node updates)
constexpr uint32_t NODE_KERNEL_BLOCK = 128;
constexpr uint32_t ELEMENT_BLOCK = 256;

// Layouts of the push_constant blocks of Shaders/ldpc_*_batched.comp.
struct ClipPushConstants
{
    uint32_t n, in_stride, out_stride;
};
struct CnPushConstants
{
    uint32_t Zc, cn_stride, num_rows, first_iter, total_stride, msg_stride;
};
struct VnPushConstants
{
    uint32_t Zc, vn_stride, num_cols, num_rows, msg_stride, ch_stride, total_stride;
};
struct HardDecisionPushConstants
{
    uint32_t block_length, total_stride, bits_stride;
};

inline uint32_t
blocks_for(const uint32_t n, const uint32_t bs)
{
    return (n + bs - 1) / bs;
}

std::tuple<memory_type, compute_api, int, int>
device_memory(const int dev_id)
{
    return { memory_type::DEVICE, compute_api::VULKAN, dev_id, 0 };
}

VkBuffer
buffer_of(const int dev_id, const void* ptr)
{
    return spu::Devices_manager::get_vulkan_buffer(dev_id, static_cast<uint8_t*>(const_cast<void*>(ptr)));
}

// Uploads the four tables of the (Bg, index_list) variant and fills the pointer and stride fields of 'bg' (same
// selection as upload_tables() in Decoder_LDPC_vulkan_kernel.cpp).
void
upload_tables(Std_5G_base_graph& bg, const int dev_id)
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

    auto upload = [dev_id](const uint32_t*& dst, const void* src, const size_t bytes)
    {
        uint8_t* ptr = spu::Devices_manager::allocate_memory(bytes, device_memory(dev_id));
        // a staged copy when the allocation is not host-visible
        spu::Devices_manager::upload_vulkan_memory(dev_id, ptr, src, bytes);
        dst = reinterpret_cast<const uint32_t*>(ptr);
    };
    upload(bg.cn_degree, h_cn_degree[b][ils], sz_cn_degree[b][ils]);
    upload(bg.vn_degree, h_vn_degree[b][ils], sz_vn_degree[b][ils]);
    upload(bg.cn, h_cn[b][ils], sz_cn[b][ils]);
    upload(bg.vn, h_vn[b][ils], sz_vn[b][ils]);
    bg.cn_stride = sz_cn[b][ils] / (sizeof(uint32_t) * bg.num_rows);
    bg.vn_stride = sz_vn[b][ils] / (sizeof(uint32_t) * bg.num_cols);
}

void
deallocate(const int dev_id, const void* ptr)
{
    if (ptr != nullptr)
        spu::Devices_manager::deallocate_memory(static_cast<uint8_t*>(const_cast<void*>(ptr)), device_memory(dev_id));
}
}

sp_vulkan::Vulkan_decoder_batched::Vulkan_decoder_batched(const int dev_id, const int K, const int N, const size_t n_frames)
  : dev_id(dev_id)
  , bg(aff3ct::tools::build_5G_base_graph(K, N))
{
    upload_tables(this->bg, this->dev_id);
    this->reserve(n_frames);
}

sp_vulkan::Vulkan_decoder_batched::~Vulkan_decoder_batched()
{
    deallocate(this->dev_id, this->llr_msg);
    deallocate(this->dev_id, this->llr_total);
    for (float* ptr : this->retired)
        deallocate(this->dev_id, ptr);
    deallocate(this->dev_id, this->bg.cn_degree);
    deallocate(this->dev_id, this->bg.vn_degree);
    deallocate(this->dev_id, this->bg.cn);
    deallocate(this->dev_id, this->bg.vn);
}

void
sp_vulkan::Vulkan_decoder_batched::reserve(const size_t n_frames)
{
    if (n_frames <= this->n_frames) return;
    for (float* ptr : { this->llr_msg, this->llr_total })
    {
        if (ptr == nullptr) continue;
        if (this->launched)
            this->retired.push_back(ptr);
        else
            deallocate(this->dev_id, ptr); // never bound to a descriptor yet
    }

    const size_t num_vns = (size_t)this->bg.num_cols * this->bg.Zc;
    const size_t num_msgs = (size_t)this->bg.num_rows * this->bg.num_cols * this->bg.Zc;
    this->llr_msg = reinterpret_cast<float*>(
      spu::Devices_manager::allocate_memory(n_frames * num_msgs * sizeof(float), device_memory(this->dev_id)));
    this->llr_total = reinterpret_cast<float*>(
      spu::Devices_manager::allocate_memory(n_frames * num_vns * sizeof(float), device_memory(this->dev_id)));
    this->n_frames = n_frames;
}

std::vector<spu::device_interface::launch_timing>
sp_vulkan::Vulkan_decoder_batched::decode(const float* llr_in,
                                          const uint32_t in_stride,
                                          int* bits,
                                          const uint32_t bits_stride,
                                          const uint32_t n_frames,
                                          const uint32_t K,
                                          const uint32_t n_ite,
                                          spu::device_interface::GpuStream* stream)
{
    // static: the address of the words is part of the executor's pipeline cache key
    static const auto clip_words = spu::sp_vulkan::spirv_words(pipeline_vulkan_shaders::ldpc_clip_channel_batched_spv,
                                                               pipeline_vulkan_shaders::ldpc_clip_channel_batched_spv_size);
    static const auto cn_words = spu::sp_vulkan::spirv_words(pipeline_vulkan_shaders::ldpc_update_cn_batched_spv,
                                                             pipeline_vulkan_shaders::ldpc_update_cn_batched_spv_size);
    static const auto vn_words = spu::sp_vulkan::spirv_words(pipeline_vulkan_shaders::ldpc_update_vn_batched_spv,
                                                             pipeline_vulkan_shaders::ldpc_update_vn_batched_spv_size);
    static const auto hd_words =
      spu::sp_vulkan::spirv_words(pipeline_vulkan_shaders::ldpc_hard_decision_batched_spv,
                                  pipeline_vulkan_shaders::ldpc_hard_decision_batched_spv_size);

    if (n_frames > this->n_frames) this->reserve(n_frames);
    this->launched = true;

    const uint32_t Zc = this->bg.Zc;
    const uint32_t num_vns = this->bg.num_cols * Zc;
    const uint32_t num_cns = this->bg.num_rows * Zc;
    const uint32_t msg_stride = this->bg.num_rows * this->bg.num_cols * Zc;

    const VkBuffer in_buf = buffer_of(this->dev_id, llr_in);
    const VkBuffer bits_buf = buffer_of(this->dev_id, bits);
    const VkBuffer msg_buf = buffer_of(this->dev_id, this->llr_msg);
    const VkBuffer total_buf = buffer_of(this->dev_id, this->llr_total);
    const VkBuffer cn_buf = buffer_of(this->dev_id, this->bg.cn);
    const VkBuffer cn_degree_buf = buffer_of(this->dev_id, this->bg.cn_degree);
    const VkBuffer vn_buf = buffer_of(this->dev_id, this->bg.vn);
    const VkBuffer vn_degree_buf = buffer_of(this->dev_id, this->bg.vn_degree);

    // Referenced by address from the descriptors: they live until launch_profiled() returns, which records (or joins
    // the open batch, which copies them) before that.
    const ClipPushConstants clip_pc{ num_vns, in_stride, num_vns };
    const CnPushConstants cn_pc_first{ Zc, this->bg.cn_stride, this->bg.num_rows, 1u, in_stride, msg_stride };
    const CnPushConstants cn_pc_rest{ Zc, this->bg.cn_stride, this->bg.num_rows, 0u, num_vns, msg_stride };
    const VnPushConstants vn_pc{ Zc, this->bg.vn_stride, this->bg.num_cols, this->bg.num_rows,
                                 msg_stride, in_stride,          num_vns };
    // after at least one iteration the totals, otherwise the channel LLRs, as 'llr_total' in the CUDA chain
    const HardDecisionPushConstants hd_pc{ K, n_ite > 0 ? num_vns : in_stride, bits_stride };

    std::vector<spu::executor::VULKAN_dispatch_desc> chain;
    chain.reserve(2 + 2 * (size_t)n_ite);

    // kept as in the original chain: iteration 0 reads the raw channel LLRs, not the clipped ones
    chain.push_back({ clip_words.data(),
                      clip_words.size() * sizeof(uint32_t),
                      { in_buf, total_buf },
                      blocks_for(num_vns, ELEMENT_BLOCK),
                      n_frames,
                      &clip_pc,
                      sizeof(clip_pc) });

    VkBuffer llr_total = in_buf;
    for (uint32_t iter = 0; iter < n_ite; ++iter)
    {
        chain.push_back({ cn_words.data(),
                          cn_words.size() * sizeof(uint32_t),
                          { llr_total, msg_buf, cn_buf, cn_degree_buf },
                          blocks_for(num_cns, NODE_KERNEL_BLOCK),
                          n_frames,
                          iter == 0 ? &cn_pc_first : &cn_pc_rest,
                          sizeof(CnPushConstants) });
        chain.push_back({ vn_words.data(),
                          vn_words.size() * sizeof(uint32_t),
                          { msg_buf, in_buf, total_buf, vn_buf, vn_degree_buf },
                          blocks_for(num_vns, NODE_KERNEL_BLOCK),
                          n_frames,
                          &vn_pc,
                          sizeof(vn_pc) });
        llr_total = total_buf;
    }

    chain.push_back({ hd_words.data(),
                      hd_words.size() * sizeof(uint32_t),
                      { llr_total, bits_buf },
                      blocks_for(K, ELEMENT_BLOCK),
                      n_frames,
                      &hd_pc,
                      sizeof(hd_pc) });

    auto* vulkan_stream = static_cast<spu::sp_vulkan::VulkanStream*>(stream);
    spu::executor::VULKAN_executor exec(vulkan_stream->device());
    exec.set_stream(vulkan_stream);
    exec.launch_profiled(chain);
    return exec.get_timings();
}

#endif // DECODER_VULKAN
