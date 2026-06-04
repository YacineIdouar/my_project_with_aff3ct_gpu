#if defined(DECODER_CUDA)
#include <algorithm>
#include <string>

#include "Module/Channel/Channel_AWGN_LLR_gpu.hpp"

using namespace aff3ct;
using namespace aff3ct::module;



template <typename R>
Channel_AWGN_LLR_gpu<R>
::Channel_AWGN_LLR_gpu(const int N, const size_t seed)
: spu::module::Stateful(), N(N), seed(seed), noised_data(this->N * this->n_frames, 0)
{
	const std::string name = "Channel_gpu";
	this->set_name(name);

	if (N <= 0)
	{
		std::stringstream message;
		message << "'N' has to be greater than 0 ('N' = " << N << ").";
		throw spu::tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
	}

	if (N % 2)
	{
		std::stringstream message;
		message << "'N' has to be divisible by 2 ('N' = " << N << ").";
		throw spu::tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
	}

	auto &p1 = this->create_task("add_noise_gpu");
	auto p1s_CP  = this->template create_socket_in <float>(p1, "CP",        1);
	auto p1s_X_N = this->template create_socket_in <R    >(p1, "X_N", this->N);
	auto p1s_Y_N = this->template create_socket_out<R    >(p1, "Y_N", this->N);

	// Setting GPU task
	size_t p1_cuda_stream = this->create_gpu_stream(p1, spu::device_interface::compute_api::CUDA, 0, 0);
	this->cuda_handler = new Cuda_channel(0);
	this->cuda_handler->init_rand_state(this->N, this->seed);

	this->register_codelet(p1, [p1s_CP, p1s_X_N, p1s_Y_N, p1_cuda_stream](Module &m, spu::runtime::Task &t, const size_t frame_id) -> int
	{
		auto &chn2 = static_cast<Channel_AWGN_LLR_gpu<R>&>(m);

		chn2.cuda_handler->add_noise(static_cast<const float*>(t[p1s_X_N ].get_dataptr()),
		                              static_cast<float*>(t[p1s_Y_N].get_dataptr()),
		                              chn2.get_N(),
		                              *static_cast<const float*>(t[p1s_CP].get_dataptr()),
		                               t.get_gpu_stream(p1_cuda_stream));

		return spu::runtime::status_t::SUCCESS;
	}, spu::device_interface::compute_api::CUDA);
}

template <typename R>
Channel_AWGN_LLR_gpu<R>* Channel_AWGN_LLR_gpu<R>
::clone() const
{
	auto m = new Channel_AWGN_LLR_gpu(*this);
	m->deep_copy(*this);
	return m;
}

template <typename R>
int Channel_AWGN_LLR_gpu<R>
::get_N() const
{
	return this->N;
}

template <typename R>
void Channel_AWGN_LLR_gpu<R>
::set_seed(const int seed)
{
	this->seed = seed;
	this->cuda_handler->init_rand_state(this->N, this->seed);
}

template <typename R>
const std::vector<R>& Channel_AWGN_LLR_gpu<R>
::get_noised_data() const
{
	return this->noised_data;
}

template<typename R>
void Channel_AWGN_LLR_gpu<R>
::set_n_frames(const size_t n_frames)
{
	const auto old_n_frames = this->get_n_frames();
	if (old_n_frames != n_frames)
	{
		Module::set_n_frames(n_frames);

		const auto old_noised_data_size = this->noised_data.size();
		const auto new_noised_data_size = (old_noised_data_size / old_n_frames) * n_frames;
		this->noised_data.resize(new_noised_data_size);
	}
}


template <typename R>
void Channel_AWGN_LLR_gpu<R>
::_add_noise(const float *CP, const R *X_N, R *Y_N, const size_t frame_id)
{
	//add_noise(static_cast<const float*>(X_N), static_cast<float*>(Y_N), this->N, *CP, this->cuda_handler->get_stream());
}



// ==================================================================================== explicit template instantiation
template class aff3ct::module::Channel_AWGN_LLR_gpu<float>;
//template class aff3ct::module::Channel_AWGN_LLR_gpu<double>;
// ==================================================================================== explicit template instantiation

#endif