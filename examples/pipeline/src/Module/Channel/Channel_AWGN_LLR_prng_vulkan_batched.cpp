// Vulkan-only module: its Vulkan part is in src/vulkan, only compiled into obj-vulkan, hence the DECODER_VULKAN guard.
#ifdef DECODER_VULKAN

#include <memory>
#include <sstream>
#include <string>

#include "Module/Channel/Channel_AWGN_LLR_prng_vulkan_batched.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Channel_AWGN_LLR_prng_vulkan_batched::Channel_AWGN_LLR_prng_vulkan_batched(const int N,
                                                                           const uint64_t seed,
                                                                           const int device_id)
  : Generic("Channel_prng_vulkan_batched")
  , N(N)
  , seed(seed)
  , device_id(device_id)
{
    if (N <= 0)
    {
        std::stringstream message;
        message << "'N' has to be greater than 0 ('N' = " << N << ").";
        throw spu::tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }

    task("add_noise_gpu", in<float>("CP", 1), in<float>("X_N", N), out<float>("Y_N", N))
      .gpu_batched(api::VULKAN, &Channel_AWGN_LLR_prng_vulkan_batched::add_noise_vulkan);

    // A replica gets no counter (per_device) and builds its own in on_clone(); main's Interface_set_seed loop then
    // gives every replica its own seed.
    replicable(true);

    // on this thread, not the stage's (see the header)
    this->dev = std::make_unique<sp_vulkan::Vulkan_channel_batched>(this->device_id, this->seed);
}

int
Channel_AWGN_LLR_prng_vulkan_batched::get_N() const
{
    return this->N;
}

void
Channel_AWGN_LLR_prng_vulkan_batched::on_clone(const size_t /*replica_id*/)
{
    this->dev = std::make_unique<sp_vulkan::Vulkan_channel_batched>(this->device_id, this->seed);
}

void
Channel_AWGN_LLR_prng_vulkan_batched::on_device(const api exec_api, const int device_id, const int /*platform_id*/)
{
    if (exec_api != api::VULKAN || (this->dev && device_id == this->device_id)) return;
    this->device_id = device_id;
    this->dev = std::make_unique<sp_vulkan::Vulkan_channel_batched>(device_id, this->seed);
}

void
Channel_AWGN_LLR_prng_vulkan_batched::set_seed(const int seed)
{
    this->seed = (uint64_t)(uint32_t)seed;
    if (this->dev) this->dev->set_seed(this->seed);
}

std::vector<spu::device_interface::launch_timing>
Channel_AWGN_LLR_prng_vulkan_batched::add_noise_vulkan(span<const float> CP,
                                                       span<const float> X_N,
                                                       span<float> Y_N,
                                                       call c)
{
    const float sigma = CP[0];
    for (size_t f = 1; f < CP.size(); f++)
        if (CP[f] != sigma)
        {
            std::stringstream message;
            message << "The frames of a firing must share the same sigma (frame 0: " << sigma << ", frame " << f
                    << ": " << CP[f] << ").";
            throw spu::tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
        }

    return this->dev->add_noise(X_N.data(), Y_N.data(), X_N.size(), sigma, c.stream);
}

#endif // DECODER_VULKAN
