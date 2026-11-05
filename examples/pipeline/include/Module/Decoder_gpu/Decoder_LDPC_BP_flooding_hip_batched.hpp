#ifndef DECODER_LDPC_BP_FLOODING_HIP_BATCHED_HPP_
#define DECODER_LDPC_BP_FLOODING_HIP_BATCHED_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <streampu.hpp>

#include "Hip/Decoder_LDPC_hip_batched_kernel.hpp"

namespace aff3ct
{
namespace module
{
/**
 * 5G LDPC BP flooding (offset min-sum) decoder on HIP (port of Decoder_LDPC_BP_flooding_gpu's HIP backend) written
 * with TidePU's Generic module: same kernels and arithmetic, so the same decoded bits. The HIP counterpart of
 * Decoder_LDPC_BP_flooding_cuda_batched.
 *
 * Same task and sockets ('decode_siho_gpu': 'Y_N', 'CWD', 'V_K'). The frames of a firing are decoded independently,
 * so the implementation is registered with 'gpu_batched()': one chain of kernels decodes the n_f frames, with the
 * frame strides given by the call ('call::frame_elmts()'). Two variants, with the same decoded bits:
 * - batched (default): each kernel has the frame as its grid y dimension (n_f times the threads of one frame);
 * - frame loop ('frame_loop' = true): each kernel has the grid of one frame, and each thread loops over the n_f
 *   frames. It pays off when one frame already fills the GPU.
 *
 * The base graph tables and the work buffers of n_f frames are device state of each replica ('per_device'): created
 * by on_device(), re-sized by on_n_frames().
 *
 * While a stream capture is in flight anywhere in the process (another replica recording its HIP graph), the HIP
 * runtime fails a synchronous hipMemcpy() with hipErrorStreamCaptureImplicit and invalidates every capture in flight,
 * whatever its mode ("operation failed due to a previous error during capture" in the replica that was recording). So
 * the setup uploads the tables on a stream of its own instead, and a replica builds its state when it is cloned
 * (on_clone()), ahead of the run, rather than lazily at its first firing as StreamPU would.
 */
class Decoder_LDPC_BP_flooding_hip_batched final : public spu::module::Generic<Decoder_LDPC_BP_flooding_hip_batched>
{
  private:
    const int K;      // information bits per frame (size of 'V_K')
    const int N_LDPC; // LLRs per frame at the input (size of 'Y_N', the depunctured codeword)
    const int N_cw;   // transmitted length: selects the 5G base graph with K
    const int n_ite;
    const bool frame_loop; // frame-loop kernels instead of the frames on grid y
    spu::module::per_device<sp_hip::Hip_decoder_batched> dev;

  public:
    Decoder_LDPC_BP_flooding_hip_batched(const int K,
                                         const int N_LDPC,
                                         const int N_cw,
                                         const int n_ite,
                                         const bool frame_loop = false);

    void on_device(const api exec_api, const int device_id, const int platform_id) override;
    void on_n_frames(const size_t n_frames) override;
    void on_clone(const size_t replica_id) override;

    // The n_f frames of the firing in one chain. 'CWD' is not written, as in the original decoder.
    std::vector<launch_timing> decode_hip(span<const float> Y_N, span<int8_t> CWD, span<int> V_K, call c);
};
}
}

#endif /* DECODER_LDPC_BP_FLOODING_HIP_BATCHED_HPP_ */
