#ifndef Channel_AWGN_LLR_GPU_HPP_
#define Channel_AWGN_LLR_GPU_HPP_

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

#include <aff3ct.hpp>
#include "Cuda/Channel_AWGN_LLR_cuda.hpp"

namespace aff3ct
{
namespace module
{
	namespace chn2
	{
		enum class tsk : size_t { add_noise, SIZE };

		namespace sck
		{
			enum class add_noise  : size_t { CP, X_N, Y_N, status };
		}
	}

template <typename R = float>
class Channel_AWGN_LLR_gpu : public spu::module::Stateful, public spu::tools::Interface_set_seed
{
public:
	inline spu::runtime::Task&   operator[](const chn2::tsk             t);
	inline spu::runtime::Socket& operator[](const chn2::sck::add_noise  s);
	inline spu::runtime::Socket& operator[](const std::string& tsk_sck);

protected:
	const int N;                 // Size of one frame (= number of bits in one frame)
	size_t total_size;
	int seed;
	std::vector<R> noised_data;  // vector of the noise applied to the signal
	Cuda_channel* cuda_handler;
	int dev_id, platform_id;

public:
	Channel_AWGN_LLR_gpu(const int N, size_t seed = 42, const int dev_id = 0, const int platform_id = 0);

	virtual ~Channel_AWGN_LLR_gpu() = default;

	virtual Channel_AWGN_LLR_gpu<R>* clone() const;

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
#include "Module/Channel/Channel_AWGN_LLR_gpu.hxx"
#endif

#endif /* Channel_AWGN_LLR_GPU_HPP_ */
