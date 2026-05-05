#include <algorithm>
#include <mipp.h>
#include <sstream>
#include <streampu.hpp>
#include <string>

#include "Decoder_LDPC_BP_flooding_cuda.hpp"
#include "Tools/Perf/common/hard_decide.h"
#include "Tools/general_utils.h"

namespace aff3ct
{
namespace module
{
template<typename B, typename R>
Decoder_LDPC_BP_flooding_cuda<B, R>::Decoder_LDPC_BP_flooding_cuda(
  const int K, const int N_LDPC, const int N_cw, const int nb_iter)
  : Decoder(K, N_LDPC), n_ite(nb_iter), N_cw(N_cw)
{
    const std::string name = "Decoder_LDPC_BP_flooding_cuda";
    this->set_name(name);

	this->ctx = sp_cuda::ldpc_decoder_init(this->K, this->N_cw, 1);

	auto& p1 = this->create_task("decode_siho_cuda");
    auto p1s_Y_N = this->template create_socket_in<B>(p1, "Y_N", this->N);
    auto p1s_CWD = this->template create_socket_out<int16_t>(p1, "CWD", 1);
    auto p1s_V_K = this->template create_socket_out<R>(p1, "V_K", this->K);

	// Enable GPU
	p1.set_compute_api(spu::device_interface::compute_api::CUDA);
	p1.set_execution_device_id(0);
	p1.set_execution_platform_id(0);


    this->create_codelet(
      p1,
      [p1s_Y_N, p1s_CWD, p1s_V_K](spu::module::Module& m, spu::runtime::Task& t, const size_t frame_id) -> int
      {
          auto& dec = static_cast<Decoder_LDPC_BP_flooding_cuda<B, R>&>(m);

          dec._decode_siho_gpu(static_cast<B*>(t[p1s_Y_N].get_dataptr()),
                            	static_cast<int16_t*>(t[p1s_CWD].get_dataptr()),
                            	static_cast<R*>(t[p1s_V_K].get_dataptr()),
                            	frame_id);

          return spu::runtime::status_t::SUCCESS;
      });
}

template<typename B, typename R>
Decoder_LDPC_BP_flooding_cuda<B, R>*
Decoder_LDPC_BP_flooding_cuda<B, R>::clone() const
{
	auto m = new Decoder_LDPC_BP_flooding_cuda<B, R>(*this);
	m->deep_copy(*this);
	return m;

}

template<typename B, typename R>
int
Decoder_LDPC_BP_flooding_cuda<B, R>::_decode_siho_gpu(const B* Y_N, int16_t* CWD, R* V_K, const size_t frame_id)
{
    sp_cuda::ldpc_decode(this->ctx,static_cast<const int16_t*>(Y_N),
				this->K,
				this->n_ite,
				0,
                static_cast<int*>(V_K));
	return 0;
}

template<typename B, typename R>
inline spu::runtime::Task&
Decoder_LDPC_BP_flooding_cuda<B, R>::operator[](const dec_cuda::tsk t)
{
	return spu::module::Module::operator[]((size_t)t);
}

template<typename B, typename R>
inline spu::runtime::Socket& 
Decoder_LDPC_BP_flooding_cuda<B, R>::operator[](const dec_cuda::sck::dec_siho_cuda s)
{
	return spu::module::Module::operator[]((size_t)dec_cuda::tsk::dec_siho_cuda)[(size_t)s];
}

template<typename B, typename R>
inline spu::runtime::Socket&
Decoder_LDPC_BP_flooding_cuda<B, R>::operator[](const std::string& tsk_sck)
{
	return spu::module::Module::operator[](tsk_sck);
}
}
}

