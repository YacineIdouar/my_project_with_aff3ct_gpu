#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"

namespace sp_cuda
{
/**
 * State of Decoder_LDPC_BP_flooding_cuda_batched on one device: the base graph tables and the work buffers of n_f
 * frames (the Cuda_decoder of Decoder_LDPC_cuda_kernel.hpp, with the frames of a firing decoded in one chain).
 *
 * Two variants of the same chain, with the same decoded bits:
 * - batched (default): each kernel has the frames on grid y, so n_f times the threads of one frame;
 * - frame loop: each kernel has the grid of one frame, and each thread loops over the n_f frames.
 *
 * Holds no CUDA type, so that nvcc and the host compiler agree on the class layout.
 */
class Cuda_decoder_batched
{
  private:
    int dev_id;
    aff3ct::tools::Std_5G_base_graph bg; // with the device copies of its tables: cn, vn, cn_degree, vn_degree
    bool frame_loop;                     // frame-loop kernels instead of the frames on grid y
    float* llr_msg = nullptr;            // n_f frames of num_rows * num_cols * Zc messages
    float* llr_total = nullptr;          // n_f frames of num_cols * Zc totals
    size_t n_frames = 0;                 // frames the buffers hold

  public:
    // Base graph of the (K, N) 5G code ('N': transmitted length), uploaded to device 'dev_id'.
    Cuda_decoder_batched(int dev_id, int K, int N, size_t n_frames, bool frame_loop = false);
    ~Cuda_decoder_batched();
    Cuda_decoder_batched(const Cuda_decoder_batched&) = delete;
    Cuda_decoder_batched& operator=(const Cuda_decoder_batched&) = delete;

    const aff3ct::tools::Std_5G_base_graph& get_base_graph() const { return this->bg; }

    // Work buffers for 'n_frames' frames (re-allocated when it changes).
    void reserve(size_t n_frames);

    bool is_frame_loop() const { return this->frame_loop; }

    // Decodes 'n_frames' frames in one chain of kernels: frame f reads 'llr_in + f * in_stride' (num_cols * Zc LLRs)
    // and writes 'bits + f * bits_stride' (K bits).
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
