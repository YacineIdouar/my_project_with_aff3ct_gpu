#pragma once

#include <cstdint>
#include <streampu.hpp>
#include "Module/Decoder_gpu/gpu_decoder_profiling.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg1.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg2.hpp"


namespace spu { namespace executor { class SYCL_executor; } }

typedef float llr_msg_t;
typedef float llr_accumulator_t;

namespace sp_sycl
{
	struct ThreadContext {
		float* llr_in_buffer = nullptr;
		int* llr_bits_out_buffer = nullptr;
		uint32_t* syndrome_buffer = nullptr;
		llr_msg_t* llr_msg_buffer = nullptr;
		llr_accumulator_t* llr_total_buffer = nullptr;
	};

	class Sycl_decoder
	{
	private:
		int dev_id;
		int platform_id;
		spu::executor::SYCL_executor* executor;
		// Per-decoder profiling totals, registered process-wide; see
		// Module/Decoder_gpu/gpu_decoder_profiling.hpp.
		gpu_prof::accumulator prof{ "SYCL", /*gpu_timed=*/false };
		ThreadContext* context;
	public:
		Sycl_decoder(int device_id, int platform_id = 0);
		void ldpc_decoder_init(int K, int N);
		void ldpc_decoder_init_context();

		uint32_t ldpc_decode(
			float const* llr_in,            // g_bg.num_cols * g_bg.Zc bytes
			uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
			uint32_t       num_iter,
			uint32_t       perform_syndrome_check,
			int* llr_bits_out,
			spu::device_interface::GpuStream* stream);

		void ldpc_decoder_shutdown();
	};

}
