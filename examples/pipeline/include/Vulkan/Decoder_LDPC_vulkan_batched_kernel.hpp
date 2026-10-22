#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"

namespace sp_vulkan
{
/**
 * State of Decoder_LDPC_BP_flooding_vulkan_batched on one device: the base graph tables and the work buffers of n_f
 * frames (the Vulkan decoder of Decoder_LDPC_vulkan_kernel.hpp, with the frames of a firing decoded in one chain, as
 * sp_cuda::Cuda_decoder_batched does on CUDA).
 *
 * Unlike the per-frame Vulkan decoder, the base graph belongs to the object rather than to a file-scope global, so two
 * decoders of different codes can coexist.
 *
 * Holds no Vulkan type, so that the module's translation unit (obj-module) does not need vulkan.h.
 */
class Vulkan_decoder_batched
{
  private:
    int dev_id;
    aff3ct::tools::Std_5G_base_graph bg; // with the device copies of its tables: cn, vn, cn_degree, vn_degree
    float* llr_msg = nullptr;            // n_f frames of num_rows * num_cols * Zc messages
    float* llr_total = nullptr;          // n_f frames of num_cols * Zc totals
    size_t n_frames = 0;                 // frames the buffers hold
    bool launched = false;               // decode() has been called: a stream may hold recordings of the buffers
    // Work buffers replaced by a larger pair after a launch, freed with the object. Not freed on the spot: the stream
    // keeps the command buffers it recorded with them, keyed on their VkBuffer handles, and a new buffer given a
    // freed handle would replay one of those recordings onto the freed memory.
    std::vector<float*> retired;

  public:
    // Base graph of the (K, N) 5G code ('N': transmitted length), uploaded to device 'dev_id'.
    Vulkan_decoder_batched(int dev_id, int K, int N, size_t n_frames);
    ~Vulkan_decoder_batched();
    Vulkan_decoder_batched(const Vulkan_decoder_batched&) = delete;
    Vulkan_decoder_batched& operator=(const Vulkan_decoder_batched&) = delete;

    const aff3ct::tools::Std_5G_base_graph& get_base_graph() const { return this->bg; }

    // Work buffers for at least 'n_frames' frames (only re-allocated to grow, see 'retired').
    void reserve(size_t n_frames);

    // Decodes 'n_frames' frames in one chain of dispatches: frame f reads 'llr_in + f * in_stride' (num_cols * Zc
    // LLRs) and writes 'bits + f * bits_stride' (K bits). 'llr_in' and 'bits' are the base pointers of their
    // allocations (the sockets'), as Vulkan addresses a buffer, not a pointer into it.
    std::vector<spu::device_interface::launch_timing> decode(const float* llr_in,
                                                             uint32_t in_stride,
                                                             int* bits,
                                                             uint32_t bits_stride,
                                                             uint32_t n_frames,
                                                             uint32_t K,
                                                             uint32_t n_ite,
                                                             spu::device_interface::GpuStream* stream);
};
}
