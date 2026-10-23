// Checks the TidePU ports against the original modules of this example, on the same inputs, for each compiled backend:
// - CUDA: aff3ct::module::Channel_AWGN_LLR_curand_batched vs Channel_AWGN_LLR_gpu (cuRAND): same seed, same states,
//   same kernel, so the same noisy frames, at 1 and 4 frames per firing;
//   aff3ct::module::Decoder_LDPC_BP_flooding_cuda_batched (n_f frames per launch) vs Decoder_LDPC_BP_flooding_gpu (one
//   frame per launch, CUDA): same decoded bits for every frame.
// - Vulkan: aff3ct::module::Channel_AWGN_LLR_prng_vulkan_batched vs Channel_AWGN_LLR_prng_gpu (Philox, VULKAN codelet):
//   same seed, same counter sequence, same dispatch, so the same noisy frames, at 1 and 4 frames per firing;
//   aff3ct::module::Decoder_LDPC_BP_flooding_vulkan_batched (n_f frames per chain) vs Decoder_LDPC_BP_flooding_gpu
//   (one frame per chain, VULKAN): same decoded bits for every frame.
// - Both backends: the frame-loop variant of the batched decoder (the threads of one frame, each looping over the n_f
//   frames) vs the same original decoder: same decoded bits for every frame.
//
// Usage: pipeline_tidepu_check [K] [N] [n_ite]   (5G LDPC, default K=1000 N=2000 n_ite=10)

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <aff3ct.hpp>

#include "Module/Channel/Channel_AWGN_LLR_gpu.hpp"
#include "Module/Channel/Channel_AWGN_LLR_prng_gpu.hpp"
#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_gpu.hpp"
#ifdef DECODER_CUDA
#include "Module/Channel/Channel_AWGN_LLR_curand_batched.hpp"
#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_cuda_batched.hpp"
#endif
#ifdef DECODER_VULKAN
#include "Module/Channel/Channel_AWGN_LLR_prng_vulkan_batched.hpp"
#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_vulkan_batched.hpp"
#endif

using api = spu::device_interface::compute_api;
namespace gm = spu::module;

namespace
{
constexpr size_t N_FRAMES_TOTAL = 8;

int n_failures = 0;

void
check(const bool condition, const std::string& what)
{
    std::cout << (condition ? "  [ok]   " : "  [FAIL] ") << what << std::endl;
    if (!condition) n_failures++;
}

// Plays precomputed frames (and one sigma per frame) into a sequence.
class Player final : public gm::Generic<Player>
{
    std::vector<std::vector<float>> frames;
    float sigma;
    size_t next = 0;
    size_t n;

  public:
    Player(const std::vector<std::vector<float>>& frames, const float sigma)
      : Generic("Player")
      , frames(frames)
      , sigma(sigma)
      , n(frames[0].size())
    {
        task("play", out<float>("x", n), out<float>("sigma", 1)).cpu(&Player::play);
    }
    void play(float* x, float* s)
    {
        std::copy(this->frames[this->next].begin(), this->frames[this->next].end(), x);
        *s = this->sigma;
        this->next++;
    }
    bool is_done() const override { return this->next >= this->frames.size(); }
};

template<typename T>
class Recorder final : public gm::Generic<Recorder<T>>
{
    using base = gm::Generic<Recorder<T>>;
    size_t n;

  public:
    std::vector<std::vector<T>> frames;
    explicit Recorder(const size_t n)
      : base("Recorder")
      , n(n)
    {
        this->task("record", base::template in<T>("y", n)).cpu(&Recorder::record);
    }
    void record(const T* y) { this->frames.emplace_back(y, y + this->n); }
};

// Runs 'module' (task 'task_name', input socket 'in', output socket 'out', optional sigma socket 'cp') on 'frames'
// with 'n_frames' frames per firing, and returns the output frames.
template<typename T>
std::vector<std::vector<T>>
run(spu::module::Module& module,
    const std::string& task_name,
    const std::string& in,
    const std::string& out,
    const std::string& cp,
    const size_t out_elmts,
    const std::vector<std::vector<float>>& frames,
    const float sigma,
    const size_t n_frames)
{
    Player player(frames, sigma);
    Recorder<T> recorder(out_elmts);
    module[task_name + "::" + in] = player["play::x"];
    if (!cp.empty()) module[task_name + "::" + cp] = player["play::sigma"];
    recorder["record::y"] = module[task_name + "::" + out];

    spu::runtime::Sequence seq(player("play"), 1);
    seq.set_n_frames(n_frames);
    seq.exec();
    return recorder.frames;
}

// Builds a module ready to run on its backend, from the size of its frames (and a seed for a channel).
using make_channel = std::function<std::unique_ptr<spu::module::Module>(int N, int seed)>;
using make_decoder = std::function<std::unique_ptr<spu::module::Module>()>;

// Selects the (per-frame) codelet of 'api' on the single task of an original module.
template<class M>
std::unique_ptr<spu::module::Module>
on_api(std::unique_ptr<M> module, const std::string& task_name, const api exec_api)
{
    (*module)(task_name).set_execution_device_info({ exec_api, 0, 0 }, true);
    return module;
}

// Switches a Generic module to 'api'.
template<class M>
std::unique_ptr<spu::module::Module>
use_api(std::unique_ptr<M> module, const api exec_api)
{
    module->use(exec_api, 0);
    return module;
}

void
test_channel(const std::string& what, const int N, const make_channel& make_ref, const make_channel& make_new)
{
    std::vector<std::vector<float>> frames(N_FRAMES_TOTAL, std::vector<float>(N));
    for (size_t f = 0; f < N_FRAMES_TOTAL; f++)
        for (int i = 0; i < N; i++)
            frames[f][i] = ((f + i) % 2) ? 1.f : -1.f;
    const float sigma = 0.7f;
    const int seed = 1234;

    for (const size_t n_frames : { 1, 4 })
    {
        std::cout << "# Channel: " << what << ", " << n_frames << " frame(s)/firing" << std::endl;
        const auto ref_chn = make_ref(N, seed);
        const auto new_chn = make_new(N, seed);

        const auto ref = run<float>(*ref_chn, "add_noise_gpu", "X_N", "Y_N", "CP", N, frames, sigma, n_frames);
        const auto got = run<float>(*new_chn, "add_noise_gpu", "X_N", "Y_N", "CP", N, frames, sigma, n_frames);
        check(ref.size() == N_FRAMES_TOTAL && got == ref, "same noisy frames as the original module");

        // and the noise is a standard Gaussian scaled by sigma
        std::vector<std::vector<double>> noise(got.size(), std::vector<double>(N));
        double sum = 0., sum2 = 0.;
        size_t count = 0;
        for (size_t f = 0; f < got.size(); f++)
            for (int i = 0; i < N; i++)
            {
                const double z = (got[f][i] - frames[f][i]) / sigma;
                noise[f][i] = z;
                sum += z;
                sum2 += z * z;
                count++;
            }
        const double mean = sum / count, var = sum2 / count - mean * mean;
        check(std::fabs(mean) < 0.05 && std::fabs(var - 1.) < 0.05,
              "noise ~ N(0, sigma^2): mean " + std::to_string(mean) + ", variance " + std::to_string(var));

        // a generator whose state did not move on would draw the same noise at the same place of the next firing
        bool fresh = got.size() == N_FRAMES_TOTAL;
        for (size_t f = 0; fresh && f + n_frames < got.size(); f++)
            fresh = noise[f] != noise[f + n_frames];
        check(fresh, "each firing draws new noise");
    }
}

void
test_decoder(const std::string& what,
             const int K,
             const int N,
             const int n_ite,
             const float sigma,
             const make_decoder& make_ref,
             const make_decoder& make_new)
{
    const auto bg = aff3ct::tools::build_5G_base_graph(K, N);
    const int N_LDPC = (int)(bg.num_cols * bg.Zc);

    // all-zero codeword through BPSK + AWGN, as LLRs: at a high sigma the decoder fails on part of the bits, which
    // makes the comparison meaningful (identical errors, not just identical zeros)
    std::mt19937 prng(42);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<std::vector<float>> frames(N_FRAMES_TOTAL, std::vector<float>(N_LDPC));
    for (auto& frame : frames)
        for (auto& llr : frame)
            llr = 2.f * (1.f + sigma * normal(prng)) / (sigma * sigma);

    std::cout << "# Decoder: " << what << ", K=" << K << " N=" << N << " (N_LDPC=" << N_LDPC << ", BG" << bg.Bg
              << ", Zc=" << bg.Zc << "), " << n_ite << " iterations, sigma " << sigma << std::endl;
    const auto old_dec = make_ref();
    const auto ref = run<int>(*old_dec, "decode_siho_gpu", "Y_N", "V_K", "", K, frames, sigma, 1);

    size_t ref_errors = 0;
    for (const auto& frame : ref)
        for (int b : frame)
            ref_errors += (b != 0);

    for (const size_t n_frames : { 1, 4, 8 })
    {
        const auto new_dec = make_new();
        const auto got = run<int>(*new_dec, "decode_siho_gpu", "Y_N", "V_K", "", K, frames, sigma, n_frames);
        check(ref.size() == N_FRAMES_TOTAL && got == ref,
              std::to_string(n_frames) + " frame(s) per launch: same bits as the original decoder (one frame per "
                                         "launch), " +
                std::to_string(ref_errors) + " bit error(s) over " + std::to_string(N_FRAMES_TOTAL * K));
    }
}

// The batched modules of one backend, and the frame-loop variant of its batched decoder, against the original modules
// on that backend.
void
test_backend(const std::string& name,
             const int K,
             const int N,
             const int n_ite,
             const make_channel& ref_chn,
             const make_channel& new_chn,
             const make_decoder& ref_dec,
             const make_decoder& new_dec,
             const make_decoder& loop_dec)
{
    test_channel(name, N, ref_chn, new_chn);
    for (const float sigma : { 0.75f, 1.f, 1.3f, 2.f })
    {
        test_decoder(name + ", batched", K, N, n_ite, sigma, ref_dec, new_dec);
        test_decoder(name + ", frame loop", K, N, n_ite, sigma, ref_dec, loop_dec);
    }
}
}

int
main(int argc, char** argv)
{
    const int K = argc > 1 ? std::atoi(argv[1]) : 1000;
    const int N = argc > 2 ? std::atoi(argv[2]) : 2 * K;
    const int n_ite = argc > 3 ? std::atoi(argv[3]) : 10;
    const auto bg = aff3ct::tools::build_5G_base_graph(K, N);
    const int N_LDPC = (int)(bg.num_cols * bg.Zc);

    spu::Devices_manager::explore_gpu_devices();

#ifdef DECODER_CUDA
    test_backend(
      "CUDA, cuRAND port vs Channel_AWGN_LLR_gpu / batched decoder vs Decoder_LDPC_BP_flooding_gpu",
      K,
      N,
      n_ite,
      [](int n, int seed)
      { return on_api(std::make_unique<aff3ct::module::Channel_AWGN_LLR_gpu<float>>(n, seed), "add_noise_gpu", api::CUDA); },
      [](int n, int seed)
      { return use_api(std::make_unique<aff3ct::module::Channel_AWGN_LLR_curand_batched>(n, seed), api::CUDA); },
      [&]()
      {
          return on_api(std::make_unique<aff3ct::module::Decoder_LDPC_BP_flooding_gpu<float, int>>(K, N_LDPC, N, n_ite, 0, 0),
                        "decode_siho_gpu",
                        api::CUDA);
      },
      [&]()
      {
          return use_api(std::make_unique<aff3ct::module::Decoder_LDPC_BP_flooding_cuda_batched>(K, N_LDPC, N, n_ite),
                         api::CUDA);
      },
      [&]()
      {
          return use_api(
            std::make_unique<aff3ct::module::Decoder_LDPC_BP_flooding_cuda_batched>(K, N_LDPC, N, n_ite, true),
            api::CUDA);
      });
#endif
#ifdef DECODER_VULKAN
    test_backend(
      "Vulkan, Philox port vs Channel_AWGN_LLR_prng_gpu / batched decoder vs Decoder_LDPC_BP_flooding_gpu",
      K,
      N,
      n_ite,
      [](int n, int seed)
      {
          return on_api(
            std::make_unique<aff3ct::module::Channel_AWGN_LLR_prng_gpu<float>>(n, seed), "add_noise_gpu", api::VULKAN);
      },
      [](int n, int seed)
      { return use_api(std::make_unique<aff3ct::module::Channel_AWGN_LLR_prng_vulkan_batched>(n, seed), api::VULKAN); },
      [&]()
      {
          return on_api(std::make_unique<aff3ct::module::Decoder_LDPC_BP_flooding_gpu<float, int>>(K, N_LDPC, N, n_ite, 0, 0),
                        "decode_siho_gpu",
                        api::VULKAN);
      },
      [&]()
      {
          return use_api(std::make_unique<aff3ct::module::Decoder_LDPC_BP_flooding_vulkan_batched>(K, N_LDPC, N, n_ite),
                         api::VULKAN);
      },
      [&]()
      {
          return use_api(
            std::make_unique<aff3ct::module::Decoder_LDPC_BP_flooding_vulkan_batched>(K, N_LDPC, N, n_ite, 0, true),
            api::VULKAN);
      });
#endif

    if (n_failures == 0)
        std::cout << "# All the checks passed." << std::endl;
    else
        std::cout << "# " << n_failures << " check(s) failed." << std::endl;
    return n_failures == 0 ? 0 : 1;
}
