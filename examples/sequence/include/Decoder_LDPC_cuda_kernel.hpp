#pragma once

#include <cstdint>
#include <aff3ct.hpp>
#include "ldpc_tables_bg1.hpp"
#include "ldpc_tables_bg2.hpp"


struct ThreadContext;
namespace sp_cuda
{

ThreadContext* ldpc_decoder_init(int K, int N, int make_stream);

uint32_t ldpc_decode(
		ThreadContext* context_,
        float const*  llr_in,            // g_bg.num_cols * g_bg.Zc bytes
        uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
        uint32_t       num_iter,
        uint32_t       perform_syndrome_check,
        int*       llr_bits_out);

void ldpc_decoder_shutdown(void);

}

