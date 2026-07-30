#include <algorithm>
#include <string>

#include "Module/Channel/Channel_AWGN_LLR_prng_gpu.hpp"

using namespace aff3ct;
using namespace aff3ct::module;

template <typename R>
Channel_AWGN_LLR_prng_gpu<R>
::Channel_AWGN_LLR_prng_gpu(const int N, const size_t seed, const int dev_id, const int platform_id)
: spu::module::Stateful(), N(N), total_size(N), seed(seed), noised_data(this->N * this->n_frames, 0)
, dev_id(dev_id), platform_id(platform_id)
{
	const std::string name = "Channel_gpu_prng";
	this->set_name(name);
	this->set_short_name(name);

	this->set_single_wave(true);

	if (N <= 0)
	{
		std::stringstream message;
		message << "'N' has to be greater than 0 ('N' = " << N << ").";
		throw spu::tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
	}

	// No "N divisible by 2" check here: the Philox kernel is bounds-checked per sample, unlike
	// the cuRAND one which always consumed curand_normal2() pairs.

	auto &p1 = this->create_task("add_noise_gpu");
	auto p1s_CP  = this->template create_socket_in <float>(p1, "CP",        1);
	auto p1s_X_N = this->template create_socket_in <R    >(p1, "X_N", this->N);
	auto p1s_Y_N = this->template create_socket_out<R    >(p1, "Y_N", this->N);

	// One codelet per compiled backend on the same task, as Decoder_LDPC_BP_flooding_gpu does:
	// the caller selects one with set_execution_device_info() (see --chn-api).
#ifdef DECODER_CUDA
	size_t p1_cuda_stream = this->create_gpu_stream(p1, spu::device_interface::compute_api::CUDA, this->dev_id, this->platform_id);
	this->cuda_handler = new Cuda_channel_prng(this->dev_id);
	this->cuda_handler->set_seed(this->seed);

	this->register_codelet(p1, [p1s_CP, p1s_X_N, p1s_Y_N, p1_cuda_stream](Module &m, spu::runtime::Task &t, const size_t frame_id) -> int
	{
		auto &chn = static_cast<Channel_AWGN_LLR_prng_gpu<R>&>(m);

		chn.cuda_handler->add_noise(static_cast<const float*>(t[p1s_X_N ].get_dataptr()),
		                            static_cast<float*>(t[p1s_Y_N].get_dataptr()),
		                            chn.total_size,
		                            *static_cast<const float*>(t[p1s_CP].get_dataptr()),
		                             t.get_gpu_stream(p1_cuda_stream));

		return spu::runtime::status_t::SUCCESS;
	}, spu::device_interface::compute_api::CUDA);
#endif
#ifdef DECODER_HIP
	size_t p1_hip_stream = this->create_gpu_stream(p1, spu::device_interface::compute_api::HIP, this->dev_id, this->platform_id);
	this->hip_handler = new sp_hip::Hip_channel_prng(this->dev_id);
	this->hip_handler->set_seed(this->seed);

	this->register_codelet(p1, [p1s_CP, p1s_X_N, p1s_Y_N, p1_hip_stream](Module &m, spu::runtime::Task &t, const size_t frame_id) -> int
	{
		auto &chn = static_cast<Channel_AWGN_LLR_prng_gpu<R>&>(m);

		chn.hip_handler->add_noise(static_cast<const float*>(t[p1s_X_N ].get_dataptr()),
		                            static_cast<float*>(t[p1s_Y_N].get_dataptr()),
		                            chn.total_size,
		                            *static_cast<const float*>(t[p1s_CP].get_dataptr()),
		                             t.get_gpu_stream(p1_hip_stream));

		return spu::runtime::status_t::SUCCESS;
	}, spu::device_interface::compute_api::HIP);
#endif
#ifdef DECODER_SYCL
	size_t p1_sycl_stream = this->create_gpu_stream(p1, spu::device_interface::compute_api::SYCL, this->dev_id, this->platform_id);
	this->sycl_handler = new sp_sycl::Sycl_channel_prng(this->dev_id, this->platform_id);
	this->sycl_handler->set_seed(this->seed);

	this->register_codelet(p1, [p1s_CP, p1s_X_N, p1s_Y_N, p1_sycl_stream](Module &m, spu::runtime::Task &t, const size_t frame_id) -> int
	{
		auto &chn = static_cast<Channel_AWGN_LLR_prng_gpu<R>&>(m);

		chn.sycl_handler->add_noise(static_cast<const float*>(t[p1s_X_N ].get_dataptr()),
		                            static_cast<float*>(t[p1s_Y_N].get_dataptr()),
		                            chn.total_size,
		                            *static_cast<const float*>(t[p1s_CP].get_dataptr()),
		                             t.get_gpu_stream(p1_sycl_stream));

		return spu::runtime::status_t::SUCCESS;
	}, spu::device_interface::compute_api::SYCL);
#endif
#ifdef DECODER_VULKAN
	size_t p1_vulkan_stream = this->create_gpu_stream(p1, spu::device_interface::compute_api::VULKAN, this->dev_id, this->platform_id);
	this->vulkan_handler = new sp_vulkan::Vulkan_channel_prng(this->dev_id);
	this->vulkan_handler->set_seed(this->seed);

	this->register_codelet(p1, [p1s_CP, p1s_X_N, p1s_Y_N, p1_vulkan_stream](Module &m, spu::runtime::Task &t, const size_t frame_id) -> int
	{
		auto &chn = static_cast<Channel_AWGN_LLR_prng_gpu<R>&>(m);

		chn.vulkan_handler->add_noise(static_cast<const float*>(t[p1s_X_N ].get_dataptr()),
		                              static_cast<float*>(t[p1s_Y_N].get_dataptr()),
		                              chn.total_size,
		                              *static_cast<const float*>(t[p1s_CP].get_dataptr()),
		                               t.get_gpu_stream(p1_vulkan_stream));

		return spu::runtime::status_t::SUCCESS;
	}, spu::device_interface::compute_api::VULKAN);
#endif
}

template <typename R>
Channel_AWGN_LLR_prng_gpu<R>* Channel_AWGN_LLR_prng_gpu<R>
::clone() const
{
	auto m = new Channel_AWGN_LLR_prng_gpu(*this);
	m->deep_copy(*this);
	return m;
}

template <typename R>
int Channel_AWGN_LLR_prng_gpu<R>
::get_N() const
{
	return this->N;
}

template <typename R>
void Channel_AWGN_LLR_prng_gpu<R>
::set_seed(const int seed)
{
	this->seed = seed;
	// Every compiled handler is re-keyed: which one actually runs is decided independently, by
	// the execution device info set on the task.
#ifdef DECODER_CUDA
	this->cuda_handler->set_seed((unsigned long long)seed);
#endif
#ifdef DECODER_HIP
	this->hip_handler->set_seed((unsigned long long)seed);
#endif
#ifdef DECODER_SYCL
	this->sycl_handler->set_seed((unsigned long long)seed);
#endif
#ifdef DECODER_VULKAN
	this->vulkan_handler->set_seed((unsigned long long)seed);
#endif
}

template <typename R>
const std::vector<R>& Channel_AWGN_LLR_prng_gpu<R>
::get_noised_data() const
{
	return this->noised_data;
}

template<typename R>
void Channel_AWGN_LLR_prng_gpu<R>
::set_n_frames(const size_t n_frames)
{
	const auto old_n_frames = this->get_n_frames();
	if (old_n_frames != n_frames)
	{
		Module::set_n_frames(n_frames);

		const auto old_noised_data_size = this->noised_data.size();
		const auto new_noised_data_size = (old_noised_data_size / old_n_frames) * n_frames;
		this->noised_data.resize(new_noised_data_size);

		// Only the sample count changes: the counter-based generator has no device-side state
		// to re-allocate, so there is nothing else to do here.
		this->total_size = this->N * n_frames;
	}
}




// ==================================================================================== explicit template instantiation
template class aff3ct::module::Channel_AWGN_LLR_prng_gpu<float>;
//template class aff3ct::module::Channel_AWGN_LLR_prng_gpu<double>;
// ==================================================================================== explicit template instantiation
