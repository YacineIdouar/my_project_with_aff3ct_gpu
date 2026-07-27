
#include <sstream>
#include <string>

#include "Module/Channel/Channel_AWGN_LLR_prng_gpu.hpp"

namespace aff3ct
{
namespace module
{

template <typename R>
spu::runtime::Task& Channel_AWGN_LLR_prng_gpu<R>::operator[](const chn_prng::tsk t)
{
	return Module::operator[]((size_t)t);
}

template <typename R>
spu::runtime::Socket& Channel_AWGN_LLR_prng_gpu<R>
::operator[](const chn_prng::sck::add_noise s)
{
	return Module::operator[]((size_t)chn_prng::tsk::add_noise)[(size_t)s];
}

template <typename R>
spu::runtime::Socket& Channel_AWGN_LLR_prng_gpu<R>
::operator[](const std::string& tsk_sck)
{
	return spu::module::Module::operator[](tsk_sck);
}

}
}
