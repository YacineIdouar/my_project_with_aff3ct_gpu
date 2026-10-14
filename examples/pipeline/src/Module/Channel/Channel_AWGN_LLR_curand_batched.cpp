// CUDA-only module: its CUDA part is in src/cuda, only compiled into obj-cuda, hence the DECODER_CUDA guard.
#ifdef DECODER_CUDA

#include <memory>
#include <sstream>
#include <string>

#include "Module/Channel/Channel_AWGN_LLR_curand_batched.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

Channel_AWGN_LLR_curand_batched::Channel_AWGN_LLR_curand_batched(const int N, const uint64_t seed)
  : Generic("Channel_curand_batched")
  , N(N)
  , seed(seed)
{
    if (N <= 0)
    {
        std::stringstream message;
        message << "'N' has to be greater than 0 ('N' = " << N << ").";
        throw spu::tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }

    task("add_noise_gpu", in<float>("CP", 1), in<float>("X_N", N), out<float>("Y_N", N))
      .gpu_batched(api::CUDA, &Channel_AWGN_LLR_curand_batched::add_noise_cuda);

    // A replica gets no cuRAND states (per_device) and builds its own in on_device(); main's Interface_set_seed loop
    // then gives every replica its own seed.
    replicable(true);
}

int
Channel_AWGN_LLR_curand_batched::get_N() const
{
    return this->N;
}

void
Channel_AWGN_LLR_curand_batched::on_device(const api exec_api, const int device_id, const int /*platform_id*/)
{
    if (exec_api == api::CUDA)
        this->dev = std::make_unique<Cuda_channel_batched>(device_id, (size_t)this->N * this->get_n_frames(), this->seed);
}

void
Channel_AWGN_LLR_curand_batched::on_n_frames(const size_t n_frames)
{
    if (this->dev) this->dev->init((size_t)this->N * n_frames, this->seed);
}

void
Channel_AWGN_LLR_curand_batched::set_seed(const int seed)
{
    this->seed = (uint64_t)(uint32_t)seed;
    if (this->dev) this->dev->init(this->dev->capacity(), this->seed);
}

std::vector<spu::device_interface::launch_timing>
Channel_AWGN_LLR_curand_batched::add_noise_cuda(span<const float> CP, span<const float> X_N, span<float> Y_N, call c)
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

    // only if the states were sized for fewer frames (on_n_frames() keeps them right)
    if (X_N.size() > this->dev->capacity()) this->dev->init(X_N.size(), this->seed);

    return this->dev->add_noise(X_N.data(), Y_N.data(), X_N.size(), sigma, c.stream);
}

#endif // DECODER_CUDA
