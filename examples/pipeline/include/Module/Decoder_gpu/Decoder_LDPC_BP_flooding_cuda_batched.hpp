#ifndef DECODER_LDPC_BP_FLOODING_CUDA_BATCHED_HPP_
#define DECODER_LDPC_BP_FLOODING_CUDA_BATCHED_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Cuda/Decoder_LDPC_cuda_batched_kernel.hpp"

namespace aff3ct
{
namespace module
{
/**
 * 5G LDPC BP flooding (offset min-sum) decoder on CUDA (port of Decoder_LDPC_BP_flooding_gpu's CUDA backend) written
 * with TidePU's Generic module: same kernels and arithmetic, so the same decoded bits.
 *
 * Same task and sockets ('decode_siho_gpu': 'Y_N', 'CWD', 'V_K'). The frames of a firing are decoded independently,
 * so the implementation is registered with 'gpu_batched()': one chain of kernels decodes the n_f frames, each kernel
 * with the frame as its grid y dimension and the frame strides given by the call ('call::frame_elmts()').
 *
 * The base graph tables and the work buffers of n_f frames are device state of each replica ('per_device'): created
 * by on_device(), re-sized by on_n_frames().
 */
class Decoder_LDPC_BP_flooding_cuda_batched final : public spu::module::Generic<Decoder_LDPC_BP_flooding_cuda_batched>
{
  private:
    const int K;      // information bits per frame (size of 'V_K')
    const int N_LDPC; // LLRs per frame at the input (size of 'Y_N', the depunctured codeword)
    const int N_cw;   // transmitted length: selects the 5G base graph with K
    const int n_ite;
    spu::module::per_device<sp_cuda::Cuda_decoder_batched> dev;

  public:
    Decoder_LDPC_BP_flooding_cuda_batched(const int K, const int N_LDPC, const int N_cw, const int n_ite);

    void on_device(const api exec_api, const int device_id, const int platform_id) override;
    void on_n_frames(const size_t n_frames) override;

    // The n_f frames of the firing in one chain. 'CWD' is not written, as in the original decoder.
    std::vector<launch_timing> decode_cuda(span<const float> Y_N, span<int8_t> CWD, span<int> V_K, call c);
};
}
}

#endif /* DECODER_LDPC_BP_FLOODING_CUDA_BATCHED_HPP_ */
