// Vulkan-only module: its Vulkan part is in src/vulkan, only compiled into obj-vulkan, hence the DECODER_VULKAN guard.
#ifdef DECODER_VULKAN

#include <memory>
#include <sstream>
#include <string>

#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_vulkan_batched.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Decoder_LDPC_BP_flooding_vulkan_batched::Decoder_LDPC_BP_flooding_vulkan_batched(const int K,
                                                                                 const int N_LDPC,
                                                                                 const int N_cw,
                                                                                 const int n_ite,
                                                                                 const int device_id)
  : Generic("Decoder_LDPC_BP_flooding_vulkan_batched")
  , K(K)
  , N_LDPC(N_LDPC)
  , N_cw(N_cw)
  , n_ite(n_ite)
  , device_id(device_id)
{
    task("decode_siho_gpu", in<float>("Y_N", N_LDPC), out<int8_t>("CWD", 1), out<int>("V_K", K))
      .gpu_batched(api::VULKAN, &Decoder_LDPC_BP_flooding_vulkan_batched::decode_vulkan);

    // nothing is carried from one frame to the next: the replicas of a stage each build their own buffers (in
    // on_clone())
    replicable(true);

    // on this thread, not the stage's (see the header)
    this->build(this->device_id);
}

void
Decoder_LDPC_BP_flooding_vulkan_batched::on_clone(const size_t /*replica_id*/)
{
    this->build(this->device_id);
}

void
Decoder_LDPC_BP_flooding_vulkan_batched::on_device(const api exec_api, const int device_id, const int /*platform_id*/)
{
    if (exec_api != api::VULKAN || (this->dev && device_id == this->device_id)) return;
    this->build(device_id);
}

void
Decoder_LDPC_BP_flooding_vulkan_batched::build(const int device_id)
{
    this->device_id = device_id;
    this->dev = std::make_unique<sp_vulkan::Vulkan_decoder_batched>(device_id, this->K, this->N_cw, this->get_n_frames());

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
Decoder_LDPC_BP_flooding_vulkan_batched::on_n_frames(const size_t n_frames)
{
    if (this->dev) this->dev->reserve(n_frames);
}

std::vector<spu::device_interface::launch_timing>
Decoder_LDPC_BP_flooding_vulkan_batched::decode_vulkan(span<const float> Y_N,
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

#endif // DECODER_VULKAN
