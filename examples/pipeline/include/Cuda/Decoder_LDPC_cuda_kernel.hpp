#pragma once

#include <cstdint>
#include <aff3ct.hpp>
#include "ldpc_tables_bg1.hpp"
#include "ldpc_tables_bg2.hpp"


namespace spu { namespace executor { class CUDA_executor; } }

typedef float llr_msg_t;
typedef float llr_accumulator_t;
namespace sp_cuda
{
	struct ThreadContext {
		float* llr_in_buffer = nullptr;
		int* llr_bits_out_buffer = nullptr;
		uint32_t* syndrome_buffer = nullptr;
		llr_msg_t* llr_msg_buffer = nullptr;
		llr_accumulator_t* llr_total_buffer = nullptr;
	};

	class Cuda_decoder
	{
	private:
		int dev_id;
		spu::executor::CUDA_executor* executor;
		ThreadContext* context;
	public:
		Cuda_decoder(int device_id);
		void ldpc_decoder_init(int K, int N);
		void ldpc_decoder_init_context();

		uint32_t ldpc_decode(
			float const* llr_in,            // g_bg.num_cols * g_bg.Zc bytes
			uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
			uint32_t       num_iter,
			uint32_t       perform_syndrome_check,
			int* llr_bits_out,
			spu::device_interface::GpuStream* stream);

		void ldpc_decoder_shutdown(void);
	};

}

