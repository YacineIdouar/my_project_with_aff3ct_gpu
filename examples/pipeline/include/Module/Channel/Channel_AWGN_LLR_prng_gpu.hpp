#ifndef Channel_AWGN_LLR_PRNG_GPU_HPP_
#define Channel_AWGN_LLR_PRNG_GPU_HPP_

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

#include <aff3ct.hpp>
#ifdef DECODER_CUDA
#include "Cuda/Channel_AWGN_LLR_prng_cuda.hpp"
#endif
#ifdef DECODER_VULKAN
#include "Vulkan/Channel_AWGN_LLR_prng_vulkan.hpp"
#endif

namespace aff3ct
{
namespace module
{
	namespace chn_prng
	{
		enum class tsk : size_t { add_noise, SIZE };

		namespace sck
		{
			enum class add_noise  : size_t { CP, X_N, Y_N, status };
		}
	}

/**
 * AWGN channel running on the GPU, drop-in alternative to Channel_AWGN_LLR_gpu.
 *
 * Identical task ("add_noise_gpu") and sockets (CP/X_N/Y_N), so the two are interchangeable
 * at bind time; the difference is that the Gaussian samples come from the hand-written
 * Philox4x32-10 generator of include/Rng/Philox4x32.hpp rather than from cuRAND.
 *
 * Like Decoder_LDPC_BP_flooding_gpu, this module registers one codelet per compiled backend
 * on its single task -- CUDA (src/cuda/Channel_AWGN_LLR_prng_cuda.cu) and Vulkan
 * (src/vulkan/Channel_AWGN_LLR_prng_vulkan.cpp) -- and the caller picks between them with
 * set_execution_device_info(), i.e. with --chn-api. Both dispatch the same generator with the
 * same geometry, so they produce bit-identical noise.
 *
 * Because that generator is counter-based, this module owns no device-side RNG state: there
 * is no init_rand_state() step, set_seed() is a plain re-key, and set_n_frames() only has to
 * resize host buffers. It also drops the "N must be even" constraint of the cuRAND version,
 * whose kernel consumed curand_normal2() pairs unconditionally.
 */
template <typename R = float>
class Channel_AWGN_LLR_prng_gpu : public spu::module::Stateful, public spu::tools::Interface_set_seed
{
public:
	inline spu::runtime::Task&   operator[](const chn_prng::tsk             t);
	inline spu::runtime::Socket& operator[](const chn_prng::sck::add_noise  s);
	inline spu::runtime::Socket& operator[](const std::string& tsk_sck);

protected:
	const int N;                 // Size of one frame (= number of bits in one frame)
	size_t total_size;
	int seed;
	std::vector<R> noised_data;  // vector of the noise applied to the signal
#ifdef DECODER_CUDA
	Cuda_channel_prng* cuda_handler;
#endif
#ifdef DECODER_VULKAN
	sp_vulkan::Vulkan_channel_prng* vulkan_handler;
#endif
	int dev_id, platform_id;

public:
	Channel_AWGN_LLR_prng_gpu(const int N, size_t seed = 42, const int dev_id = 0, const int platform_id = 0);

	virtual ~Channel_AWGN_LLR_prng_gpu() = default;

	virtual Channel_AWGN_LLR_prng_gpu<R>* clone() const;

	int get_N() const;

	const std::vector<R>& get_noised_data() const;

	virtual void set_seed(const int seed);

	virtual void set_n_frames(const size_t n_frames);

protected:
	void _add_noise(const float *CP, const R *X_N, R *Y_N, const size_t frame_id);
};
}
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include "Module/Channel/Channel_AWGN_LLR_prng_gpu.hxx"
#endif

#endif /* Channel_AWGN_LLR_PRNG_GPU_HPP_ */
