#pragma once

#include <cstdint>
#include "ldpc_tables_bg1.hpp"
#include "ldpc_tables_bg2.hpp"


struct ThreadContext;
namespace sp_cuda
{
	/*
 * ldpc_decoder_init
 *
 * Must be called once before any decode call.
 * Resolves the base graph for (K, N), uploads the matching table variant
 * to the GPU, and allocates per-thread working buffers.
 *
 * K           - number of information bits
 * N           - codeword length (coded bits)
 * make_stream - if non-zero, creates a dedicated high-priority CUDA stream
 *               for this thread; otherwise the caller must supply a stream
 *               to every ldpc_decode() call
 *
 * Returns a pointer to the thread-local context, or throws
 * std::invalid_argument (via aff3ct) on an invalid (K, N) pair.
 */
ThreadContext* ldpc_decoder_init(int K, int N, int make_stream);

/*
 * ldpc_decode
 *
 * Runs belief-propagation decoding and writes soft LLR totals.
 * The caller is responsible for hard-decision or further processing.
 *
 * context_               - context returned by ldpc_decoder_init();
 *                          if NULL a thread-local context is created lazily
 * stream                 - CUDA stream to enqueue work on;
 *                          if 0 and context_ is non-NULL, uses context_->stream
 * llr_in                 - input channel LLRs, num_cols * Zc signed bytes
 * llr_total_out          - output soft LLRs after decoding, num_cols * Zc
 *                          signed bytes (same layout as llr_in);
 *                          the caller owns this buffer (host or pinned memory)
 * K                      - info bit count used to bound the output region
 *                          (may be less than g_bg.K_LDPC due to zero-padding)
 * num_iter               - maximum number of BP iterations
 * perform_syndrome_check - if non-zero, verifies parity after decoding
 *
 * Returns num_iter-1 on success, num_iter+1 if syndrome check fails.
 */
uint32_t ldpc_decode(
		ThreadContext* context_,
        float const*  llr_in,            // g_bg.num_cols * g_bg.Zc bytes
        uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
        uint32_t       num_iter,
        uint32_t       perform_syndrome_check,
        int*       llr_bits_out);

/*
 * ldpc_decoder_shutdown
 *
 * Frees all GPU allocations (table buffers, working buffers, streams)
 * across all threads that called ldpc_decoder_init(). Call once at
 * program exit or when the decoder is no longer needed.
 */
void ldpc_decoder_shutdown(void);

}

