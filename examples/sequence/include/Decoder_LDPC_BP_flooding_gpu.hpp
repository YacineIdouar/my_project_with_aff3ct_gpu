/*!
 * \file
 * \brief Class module::Decoder_LDPC_BP_flooding.
 */
#ifndef DECODER_LDPC_BP_FLOODING_gpu_HPP_
#define DECODER_LDPC_BP_FLOODING_gpu_HPP_

#include <cstdint>
#include <vector>

#include "Module/Decoder/Decoder_SIHO.hpp"
#include "Module/Decoder/LDPC/BP/Decoder_LDPC_BP.hpp"
#include "Tools/Algo/Matrix/Sparse_matrix/Sparse_matrix.hpp"
#include "Tools/Code/LDPC/Syndrome/LDPC_syndrome.hpp"
#include "Tools/Code/LDPC/Update_rule/SPA/Update_rule_SPA.hpp"
#include "Tools/Code/LDPC/Standard/5G/5G_base_graph.hpp"
#include "Decoder_LDPC_cuda_kernel.hpp"
#include "Decoder_LDPC_hip_kernel.hpp"

#include "Module/Decoder/Decoder.hpp"

namespace aff3ct
{
namespace module
{

namespace dec_gpu
	{
		enum class tsk : size_t { dec_siho_gpu, SIZE };
		namespace sck
		{
			enum class dec_siho_gpu  : size_t { Y_N, CWD, V_K, status};
		}
	}

/*!
 * \class Decoder_LDPC_BP_flooding_gpu
 *
 * \brief A Decoder is an algorithm dedicated to find the initial sequence of information bits (before the noise).
 *
 * \tparam B: type of the bits in the Decoder.
 *
 * The Decoder takes a soft input (real numbers) and return a hard output (bits).
 */
template<typename B = int8_t, typename R = int>
class Decoder_LDPC_BP_flooding_gpu : public Decoder
{
  public:
    inline spu::runtime::Task& operator[](const dec_gpu::tsk t);
    inline spu::runtime::Socket& operator[](const dec_gpu::sck::dec_siho_gpu s);
    inline spu::runtime::Socket& operator[](const std::string& tsk_sck);

  public:
    /*!
     * \brief Constructor.
     *
     * \param K: number of information bits in the frame.
     * \param N: size of one frame.
     * \param nb_iter: number of iterations.
     */
    Decoder_LDPC_BP_flooding_gpu(const int K, const int N_LDPC, const int N_cw, const int nb_iter);

	virtual Decoder_LDPC_BP_flooding_gpu<B, R>* clone() const;


    /*!
     * \brief Destructor.
     */
    virtual ~Decoder_LDPC_BP_flooding_gpu() = default;

    //virtual Decoder_LDPC_BP_flooding_gpu<B, R>* clone() const;

    /*!
     * \brief Task method that decodes the noisy frame.
     *
     * \param Y_N: a noisy frame.
     * \param V_K: a decoded codeword (only the information bits).
     */
    template<class AR = std::allocator<R>, class AB = std::allocator<B>>
    int decode_siho_gpu(const std::vector<R, AR>& Y_N,
                    std::vector<int8_t, AB>& CWD,
                    std::vector<B, AB>& V_K,
                    const int frame_id = -1,
                    const bool managed_memory = true);

  protected:
  	ThreadContext* ctx;
	const int n_ite;
	const int N_cw;
    virtual int _decode_siho_gpu(const B* Y_N, int8_t* CWD, R* V_K, const size_t frame_id);
};
}
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include "Decoder_LDPC_BP_flooding_gpu.hxx"
#endif

#endif /* DECODER_LDPC_BP_FLOODING_gpu_HPP_ */