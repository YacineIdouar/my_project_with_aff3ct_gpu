// CUDA-only module: its CUDA part is in src/cuda, only compiled into obj-cuda, hence the DECODER_CUDA guard.
#ifdef DECODER_CUDA

#include <memory>
#include <sstream>
#include <string>

#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_cuda_batched.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Decoder_LDPC_BP_flooding_cuda_batched::Decoder_LDPC_BP_flooding_cuda_batched(const int K,
                                                                             const int N_LDPC,
                                                                             const int N_cw,
                                                                             const int n_ite)
  : Generic("Decoder_LDPC_BP_flooding_cuda_batched")
  , K(K)
  , N_LDPC(N_LDPC)
  , N_cw(N_cw)
  , n_ite(n_ite)
{
    task("decode_siho_gpu", in<float>("Y_N", N_LDPC), out<int8_t>("CWD", 1), out<int>("V_K", K))
      .gpu_batched(api::CUDA, &Decoder_LDPC_BP_flooding_cuda_batched::decode_cuda);

    // nothing is carried from one frame to the next: the replicas of a stage each build their own buffers
    replicable(true);
}

void
Decoder_LDPC_BP_flooding_cuda_batched::on_device(const api exec_api, const int device_id, const int /*platform_id*/)
{
    if (exec_api != api::CUDA) return;
    this->dev = std::make_unique<sp_cuda::Cuda_decoder_batched>(device_id, this->K, this->N_cw, this->get_n_frames());

    const auto& bg = this->dev->get_base_graph();
    if (bg.num_cols * bg.Zc > (uint32_t)this->N_LDPC)
    {
        std::stringstream message;
        message << "The base graph reads " << bg.num_cols * bg.Zc << " LLRs per frame, but 'Y_N' holds only "
                << this->N_LDPC << " ('N_LDPC').";
        throw spu::tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }
}

void
Decoder_LDPC_BP_flooding_cuda_batched::on_n_frames(const size_t n_frames)
{
    if (this->dev) this->dev->reserve(n_frames);
}

std::vector<spu::device_interface::launch_timing>
Decoder_LDPC_BP_flooding_cuda_batched::decode_cuda(span<const float> Y_N,
                                                   span<int8_t> /*CWD*/,
                                                   span<int> V_K,
                                                   call c)
{
    return this->dev->decode(Y_N.data(),
                             (uint32_t)c.frame_elmts(0),
                             V_K.data(),
                             (uint32_t)c.frame_elmts(2),
                             (uint32_t)c.n_frames,
                             (uint32_t)this->K,
                             (uint32_t)this->n_ite,
                             c.stream);
}

#endif // DECODER_CUDA
