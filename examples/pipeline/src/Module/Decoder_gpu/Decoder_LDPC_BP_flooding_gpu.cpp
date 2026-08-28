#include <algorithm>
#include <mipp.h>
#include <sstream>
#include <streampu.hpp>
#include <string>

#include "Module/Decoder_gpu/Decoder_LDPC_BP_flooding_gpu.hpp"
#include "Tools/Perf/common/hard_decide.h"
#include "Tools/general_utils.h"

namespace aff3ct
{
namespace module
{
template<typename B, typename R>
Decoder_LDPC_BP_flooding_gpu<B, R>::Decoder_LDPC_BP_flooding_gpu(
  const int K, const int N_LDPC, const int N_cw, const int nb_iter, const int dev_id, const int platform_id)
  : Decoder(K, N_LDPC), n_ite(nb_iter), N_cw(N_cw), dev_id(dev_id), platform_id(platform_id)
{
    const std::string name = "Decoder_LDPC_BP_flooding_gpu";
    this->set_name(name);

	auto& p1 = this->create_task("decode_siho_gpu");
    auto p1s_Y_N = this->template create_socket_in<B>(p1, "Y_N", this->N);
    auto p1s_CWD = this->template create_socket_out<int8_t>(p1, "CWD", 1);
    auto p1s_V_K = this->template create_socket_out<R>(p1, "V_K", this->K);

#ifdef DECODER_CUDA
	// Enable GPU
	size_t p1_stream_cuda = this->create_gpu_stream(p1, spu::device_interface::compute_api::CUDA, this->dev_id, this->platform_id);
	this->cuda_handler = new sp_cuda::Cuda_decoder(this->dev_id);
	this->cuda_handler->ldpc_decoder_init(this->K, this->N_cw);

    this->register_codelet(
      p1,
      [p1s_Y_N, p1s_CWD, p1s_V_K, p1_stream_cuda](spu::module::Module& m, spu::runtime::Task& t, const size_t frame_id) -> int
      {
          auto& dec = static_cast<Decoder_LDPC_BP_flooding_gpu<B, R>&>(m);

         dec.cuda_handler->ldpc_decode(
			static_cast<const float*>(t[p1s_Y_N].get_dataptr()),
			dec.K,
			dec.n_ite,
			0,
			static_cast<int*>(t[p1s_V_K].get_dataptr()),
			t.get_gpu_stream(p1_stream_cuda));

          return spu::runtime::status_t::SUCCESS;},
		  spu::device_interface::compute_api::CUDA);
#endif
#ifdef DECODER_HIP
	// Enable GPU
	size_t p1_stream_hip = this->create_gpu_stream(p1, spu::device_interface::compute_api::HIP, this->dev_id, this->platform_id);
	this->hip_handler = new sp_hip::Hip_decoder(this->dev_id);
	this->hip_handler->ldpc_decoder_init(this->K, this->N_cw);

    this->register_codelet(
      p1,
      [p1s_Y_N, p1s_CWD, p1s_V_K, p1_stream_hip](spu::module::Module& m, spu::runtime::Task& t, const size_t frame_id) -> int
      {
          auto& dec = static_cast<Decoder_LDPC_BP_flooding_gpu<B, R>&>(m);

         dec.hip_handler->ldpc_decode(
			static_cast<const float*>(t[p1s_Y_N].get_dataptr()),
			dec.K,
			dec.n_ite,
			0,
			static_cast<int*>(t[p1s_V_K].get_dataptr()),
			t.get_gpu_stream(p1_stream_hip));

          return spu::runtime::status_t::SUCCESS;},
		  spu::device_interface::compute_api::HIP);
#endif
#ifdef DECODER_SYCL
	// Enable GPU
	size_t p1_stream_sycl = this->create_gpu_stream(p1, spu::device_interface::compute_api::SYCL, this->dev_id, this->platform_id);
	this->sycl_handler = new sp_sycl::Sycl_decoder(this->dev_id, this->platform_id);
	this->sycl_handler->ldpc_decoder_init(this->K, this->N_cw);

    this->register_codelet(
      p1,
      [p1s_Y_N, p1s_CWD, p1s_V_K, p1_stream_sycl](spu::module::Module& m, spu::runtime::Task& t, const size_t frame_id) -> int
      {
          auto& dec = static_cast<Decoder_LDPC_BP_flooding_gpu<B, R>&>(m);

         dec.sycl_handler->ldpc_decode(
			static_cast<const float*>(t[p1s_Y_N].get_dataptr()),
			dec.K,
			dec.n_ite,
			0,
			static_cast<int*>(t[p1s_V_K].get_dataptr()),
			t.get_gpu_stream(p1_stream_sycl));

          return spu::runtime::status_t::SUCCESS;},
			  spu::device_interface::compute_api::SYCL);
#endif
#ifdef DECODER_VULKAN
	// Enable GPU
	size_t p1_stream_vulkan = this->create_gpu_stream(p1, spu::device_interface::compute_api::VULKAN, this->dev_id, this->platform_id);
	this->vulkan_handler = sp_vulkan::Vulkan_decoder::create(this->dev_id);
	this->vulkan_handler->ldpc_decoder_init(this->K, this->N_cw);

    this->register_codelet(
      p1,
      [p1s_Y_N, p1s_CWD, p1s_V_K, p1_stream_vulkan](spu::module::Module& m, spu::runtime::Task& t, const size_t frame_id) -> int
      {
          auto& dec = static_cast<Decoder_LDPC_BP_flooding_gpu<B, R>&>(m);

         dec.vulkan_handler->ldpc_decode(
			static_cast<const float*>(t[p1s_Y_N].get_dataptr()),
			dec.K,
			dec.n_ite,
			0,
			static_cast<int*>(t[p1s_V_K].get_dataptr()),
			t.get_gpu_stream(p1_stream_vulkan));

          return spu::runtime::status_t::SUCCESS;},
			  spu::device_interface::compute_api::VULKAN);
#endif
}

template<typename B, typename R>
Decoder_LDPC_BP_flooding_gpu<B, R>*
Decoder_LDPC_BP_flooding_gpu<B, R>::clone() const
{
	auto m = new Decoder_LDPC_BP_flooding_gpu<B, R>(*this);
	m->deep_copy(*this);
	// dev_id/platform_id were copied from *this by the copy constructor above: a replicated stage
	// must keep running on the device the original was configured for, not fall back to device 0.
#ifdef DECODER_CUDA
	m->cuda_handler = new sp_cuda::Cuda_decoder(m->dev_id);
	m->cuda_handler->ldpc_decoder_init(this->K, this->N_cw);
#endif
#ifdef DECODER_HIP
	m->hip_handler = new sp_hip::Hip_decoder(m->dev_id);
	m->hip_handler->ldpc_decoder_init(this->K, this->N_cw);
#endif
#ifdef DECODER_SYCL
	m->sycl_handler = new sp_sycl::Sycl_decoder(m->dev_id, m->platform_id);
	m->sycl_handler->ldpc_decoder_init(this->K, this->N_cw);
#endif
#ifdef DECODER_VULKAN
	m->vulkan_handler = sp_vulkan::Vulkan_decoder::create(m->dev_id);
	m->vulkan_handler->ldpc_decoder_init(this->K, this->N_cw);
#endif
	return m;

}

template<typename B, typename R>
int
Decoder_LDPC_BP_flooding_gpu<B, R>::_decode_siho_gpu(const B* Y_N, int8_t* CWD, R* V_K, const size_t frame_id)
{
    //sp_cuda::ldpc_decode(this->ctx,static_cast<const float*>(Y_N),
	//			this->K,
	//			this->n_ite,
	//			0,
    //            static_cast<int*>(V_K));
	//
	return 0;
}

template<typename B, typename R>
inline spu::runtime::Task&
Decoder_LDPC_BP_flooding_gpu<B, R>::operator[](const dec_gpu::tsk t)
{
	return spu::module::Module::operator[]((size_t)t);
}

template<typename B, typename R>
inline spu::runtime::Socket& 
Decoder_LDPC_BP_flooding_gpu<B, R>::operator[](const dec_gpu::sck::dec_siho_gpu s)
{
	return spu::module::Module::operator[]((size_t)dec_gpu::tsk::dec_siho_gpu)[(size_t)s];
}

template<typename B, typename R>
inline spu::runtime::Socket&
Decoder_LDPC_BP_flooding_gpu<B, R>::operator[](const std::string& tsk_sck)
{
	return spu::module::Module::operator[](tsk_sck);
}

// Explicit template instantiation for float, int
template class Decoder_LDPC_BP_flooding_gpu<float, int>;

}
}

