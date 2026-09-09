#ifndef Channel_AWGN_LLR_PRNG_GPU_HPP_
#define Channel_AWGN_LLR_PRNG_GPU_HPP_

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

#include <aff3ct.hpp>
#ifdef DECODER_CUDA
// cuRAND, not the Philox handler this module's other backends use -- see the note on the class.
#include "Cuda/Channel_AWGN_LLR_cuda.hpp"
#endif
#ifdef DECODER_HIP
#include "Hip/Channel_AWGN_LLR_prng_hip.hpp"
#endif
#ifdef DECODER_SYCL
#include "Sycl/Channel_AWGN_LLR_prng_sycl.hpp"
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
 * Like Decoder_LDPC_BP_flooding_gpu, this module registers one codelet per compiled backend on
 * its single task -- CUDA, HIP, SYCL and Vulkan -- and the caller picks between them with
 * set_execution_device_info(), i.e. with --chn-api.
 *
 * The CUDA codelet is the exception: it runs cuRAND (Cuda_channel, src/cuda/Channel_AWGN_LLR_
 * cuda.cu), the same function --chn-api CUDA selects through Channel_AWGN_LLR_gpu, rather than the
 * Philox handler in src/cuda/Channel_AWGN_LLR_prng_cuda.cu. So CUDA and CUDA_PRNG now produce the
 * same noise as each other, and the HIP/SYCL/Vulkan codelets -- still Philox -- produce noise
 * bit-identical among themselves but no longer to CUDA. Cuda_channel_prng is still compiled; it is
 * simply not reachable from --chn-api any more.
 *
 * That mixed sourcing is what shapes the lifecycle below. cuRAND keeps a device-side state buffer
 * sized to the sample count, so the CUDA path needs init_rand_state() where the counter-based
 * backends need nothing: set_seed() re-initialises it, set_n_frames() re-allocates it, and clone()
 * gives each replica its own. The counter-based backends keep ignoring all three.
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
	Cuda_channel* cuda_handler; // cuRAND, unlike the counter-based handlers below
#endif
#ifdef DECODER_HIP
	sp_hip::Hip_channel_prng* hip_handler;
#endif
#ifdef DECODER_SYCL
	sp_sycl::Sycl_channel_prng* sycl_handler;
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
