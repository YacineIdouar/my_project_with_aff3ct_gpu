// Checks the TidePU ports against the original CUDA modules of this example, on the same inputs:
// - aff3ct::module::Channel_AWGN_LLR_curand_batched vs Channel_AWGN_LLR_gpu (cuRAND): same seed, same states, same kernel,
//   so the same noisy frames, at 1 and 4 frames per firing;
// - aff3ct::module::Decoder_LDPC_BP_flooding_cuda_batched (n_f frames per launch) vs Decoder_LDPC_BP_flooding_gpu (one frame
//   per launch, CUDA): same decoded bits for every frame.
//
// Usage: pipeline_tidepu_check [K] [N] [n_ite]   (5G LDPC, default K=1000 N=2000 n_ite=10)

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <aff3ct.hpp>

#include "Module/Channel/Channel_AWGN_LLR_gpu.hpp"
#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_gpu.hpp"
#include "Module/Channel/Channel_AWGN_LLR_curand_batched.hpp"
#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_cuda_batched.hpp"

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

void
test_channel(const int N)
{
    std::vector<std::vector<float>> frames(N_FRAMES_TOTAL, std::vector<float>(N));
    for (size_t f = 0; f < N_FRAMES_TOTAL; f++)
        for (int i = 0; i < N; i++)
            frames[f][i] = ((f + i) % 2) ? 1.f : -1.f;
    const float sigma = 0.7f;
    const int seed = 1234;

    for (const size_t n_frames : { 1, 4 })
    {
        std::cout << "# Channel: cuRAND port vs Channel_AWGN_LLR_gpu, " << n_frames << " frame(s)/firing" << std::endl;
        aff3ct::module::Channel_AWGN_LLR_gpu<float> old_chn(N, seed);
        old_chn("add_noise_gpu").set_execution_device_info({ api::CUDA, 0, 0 }, true);
        aff3ct::module::Channel_AWGN_LLR_curand_batched new_chn(N, seed);
        new_chn.use(api::CUDA, 0);

        const auto ref = run<float>(old_chn, "add_noise_gpu", "X_N", "Y_N", "CP", N, frames, sigma, n_frames);
        const auto got = run<float>(new_chn, "add_noise_gpu", "X_N", "Y_N", "CP", N, frames, sigma, n_frames);
        check(ref.size() == N_FRAMES_TOTAL && got == ref, "same noisy frames as the original module");

        // and the noise is a standard Gaussian scaled by sigma
        double sum = 0., sum2 = 0.;
        size_t count = 0;
        for (size_t f = 0; f < got.size(); f++)
            for (int i = 0; i < N; i++)
            {
                const double z = (got[f][i] - frames[f][i]) / sigma;
                sum += z;
                sum2 += z * z;
                count++;
            }
        const double mean = sum / count, var = sum2 / count - mean * mean;
        check(std::fabs(mean) < 0.05 && std::fabs(var - 1.) < 0.05,
              "noise ~ N(0, sigma^2): mean " + std::to_string(mean) + ", variance " + std::to_string(var));
    }
}

void
test_decoder(const int K, const int N, const int n_ite, const float sigma)
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

    std::cout << "# Decoder: K=" << K << " N=" << N << " (N_LDPC=" << N_LDPC << ", BG" << bg.Bg << ", Zc=" << bg.Zc
              << "), " << n_ite << " iterations, sigma " << sigma << std::endl;
    aff3ct::module::Decoder_LDPC_BP_flooding_gpu<float, int> old_dec(K, N_LDPC, N, n_ite, 0, 0);
    old_dec("decode_siho_gpu").set_execution_device_info({ api::CUDA, 0, 0 }, true);
    const auto ref = run<int>(old_dec, "decode_siho_gpu", "Y_N", "V_K", "", K, frames, sigma, 1);

    size_t ref_errors = 0;
    for (const auto& frame : ref)
        for (int b : frame)
            ref_errors += (b != 0);

    for (const size_t n_frames : { 1, 4, 8 })
    {
        aff3ct::module::Decoder_LDPC_BP_flooding_cuda_batched new_dec(K, N_LDPC, N, n_ite);
        new_dec.use(api::CUDA, 0);
        const auto got = run<int>(new_dec, "decode_siho_gpu", "Y_N", "V_K", "", K, frames, sigma, n_frames);
        check(ref.size() == N_FRAMES_TOTAL && got == ref,
              std::to_string(n_frames) + " frame(s) per launch: same bits as the original decoder (one frame per "
                                         "launch), " +
                std::to_string(ref_errors) + " bit error(s) over " + std::to_string(N_FRAMES_TOTAL * K));
    }
}
}

int
main(int argc, char** argv)
{
    const int K = argc > 1 ? std::atoi(argv[1]) : 1000;
    const int N = argc > 2 ? std::atoi(argv[2]) : 2 * K;
    const int n_ite = argc > 3 ? std::atoi(argv[3]) : 10;

    spu::Devices_manager::explore_gpu_devices();
    test_channel(N);
    for (const float sigma : { 0.75f, 1.f, 1.3f })
        test_decoder(K, N, n_ite, sigma);

    if (n_failures == 0)
        std::cout << "# All the checks passed." << std::endl;
    else
        std::cout << "# " << n_failures << " check(s) failed." << std::endl;
    return n_failures == 0 ? 0 : 1;
}
