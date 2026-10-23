#ifndef DECODER_LDPC_BP_FLOODING_VULKAN_BATCHED_HPP_
#define DECODER_LDPC_BP_FLOODING_VULKAN_BATCHED_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Vulkan/Decoder_LDPC_vulkan_batched_kernel.hpp"

namespace aff3ct
{
namespace module
{
/**
 * 5G LDPC BP flooding (offset min-sum) decoder on Vulkan (port of Decoder_LDPC_BP_flooding_gpu's Vulkan backend)
 * written with TidePU's Generic module, the Vulkan counterpart of Decoder_LDPC_BP_flooding_cuda_batched: same shaders
 * and arithmetic, so the same decoded bits.
 *
 * Same task and sockets ('decode_siho_gpu': 'Y_N', 'CWD', 'V_K'). The frames of a firing are decoded independently,
 * so the implementation is registered with 'gpu_batched()': one chain of dispatches decodes the n_f frames, with the
 * frame strides given by the call ('call::frame_elmts()'). Two variants, with the same decoded bits:
 * - batched (default): each dispatch has the frame as its work group y (n_f times the invocations of one frame);
 * - frame loop ('frame_loop' = true): each dispatch has the work groups of one frame, and each invocation loops over
 *   the n_f frames. It pays off when one frame already fills the GPU.
 *
 * The base graph tables and the work buffers of n_f frames are device state of each replica ('per_device'), grown by
 * on_n_frames(). As in Channel_AWGN_LLR_prng_vulkan_batched, and for the same reason (StreamPU's Vulkan allocator is
 * not thread-safe, see there), they are built with the module -- by the constructor, and by on_clone() for a
 * replica, for the device given to the constructor -- rather than by on_device(), which runs on the stage's thread.
 */
class Decoder_LDPC_BP_flooding_vulkan_batched final
  : public spu::module::Generic<Decoder_LDPC_BP_flooding_vulkan_batched>
{
  private:
    const int K;      // information bits per frame (size of 'V_K')
    const int N_LDPC; // LLRs per frame at the input (size of 'Y_N', the depunctured codeword)
    const int N_cw;   // transmitted length: selects the 5G base graph with K
    const int n_ite;
    int device_id;         // the Vulkan device 'dev' is built for
    const bool frame_loop; // frame-loop shaders instead of the frames as work group y
    spu::module::per_device<sp_vulkan::Vulkan_decoder_batched> dev;

    void build(const int device_id);

  public:
    Decoder_LDPC_BP_flooding_vulkan_batched(const int K,
                                            const int N_LDPC,
                                            const int N_cw,
                                            const int n_ite,
                                            const int device_id = 0,
                                            const bool frame_loop = false);

    void on_clone(const size_t replica_id) override;
    void on_device(const api exec_api, const int device_id, const int platform_id) override;
    void on_n_frames(const size_t n_frames) override;

    // The n_f frames of the firing in one chain. 'CWD' is not written, as in the original decoder.
    std::vector<launch_timing> decode_vulkan(span<const float> Y_N, span<int8_t> CWD, span<int> V_K, call c);
};
}
}

#endif /* DECODER_LDPC_BP_FLOODING_VULKAN_BATCHED_HPP_ */
