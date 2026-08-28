#pragma once

#include <cstdint>
#include <string>
#include <streampu.hpp>
#include "Module/Decoder_gpu/gpu_decoder_profiling.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg1.hpp"
#include "Module/Decoder_gpu/ldpc_tables_bg2.hpp"

typedef float llr_msg_t;
typedef float llr_accumulator_t;

namespace sp_vulkan
{
	struct ThreadContext {
		llr_msg_t* llr_msg_buffer = nullptr;
		llr_accumulator_t* llr_total_buffer = nullptr;
	};

	// Which of the two interchangeable Vulkan decoder implementations to run. Both are always
	// compiled; the choice is made once at startup, not at build time.
	//
	//   CHAIN_REBUILT : src/vulkan/Decoder_LDPC_vulkan_kernel.cpp -- rebuilds the whole
	//                   VULKAN_dispatch_desc chain on every decode call.
	//   CHAIN_CACHED  : src/vulkan/Decoder_LDPC_vulkan_kernel_cached.cpp -- describes the chain
	//                   once per (socket buffers, num_iter, K) key and reuses the description, so a
	//                   steady-state frame is a hit in StreamPU's recorded-dispatch cache and
	//                   nothing else.
	//
	// They are bit-for-bit equivalent; the difference is host-side launch cost only.
	enum class decoder_impl
	{
		CHAIN_REBUILT,
		CHAIN_CACHED
	};

	class Vulkan_decoder
	{
	protected:
		int dev_id;
		ThreadContext* context;

		// Registered process-wide by its constructor, so the report main_gpu.cpp prints covers every
		// decoder without being handed one. Written only by the thread that owns this decoder.
		gpu_prof::accumulator prof{ "DECODER", "VULKAN" };

		// Only the two implementations derive from this; everyone else goes through create().
		explicit Vulkan_decoder(int device_id) : dev_id(device_id), context(nullptr) {}

	public:
		virtual ~Vulkan_decoder() = default;

		Vulkan_decoder(const Vulkan_decoder&) = delete;
		Vulkan_decoder& operator=(const Vulkan_decoder&) = delete;

		// Allocates one decoder of the given implementation, or of the process-wide default.
		// Caller owns the result.
		static Vulkan_decoder* create(int device_id, decoder_impl impl);
		static Vulkan_decoder* create(int device_id);

		// The implementation create(device_id) builds. Starts from the SPU_LDPC_VULKAN_CHAIN
		// environment variable ("rebuilt" or "0" select CHAIN_REBUILT, anything else and an unset
		// variable select CHAIN_CACHED), which main_gpu.cpp overrides from --dec-vk-chain.
		// Set it before the first decoder is built -- replicated stages create theirs while the
		// pipeline is being built, and each keeps whatever was current then.
		static void set_default_impl(decoder_impl impl);
		static decoder_impl get_default_impl();

		// "cached" / "rebuilt" <-> enum, for command lines and headers. Parsing is case
		// insensitive; on an unknown string it returns false and leaves 'out' untouched.
		static bool str_to_impl(const std::string& s, decoder_impl& out);
		static const char* impl_to_str(decoder_impl impl);

		virtual void ldpc_decoder_init(int K, int N) = 0;
		virtual void ldpc_decoder_init_context() = 0;

		virtual uint32_t ldpc_decode(
			float const* llr_in,            // g_bg.num_cols * g_bg.Zc bytes
			uint32_t       K,                  // info bits to unpack (≤ g_bg.K_LDPC)
			uint32_t       num_iter,
			uint32_t       perform_syndrome_check,
			int* llr_bits_out,
			spu::device_interface::GpuStream* stream) = 0;

		virtual void ldpc_decoder_shutdown(void) = 0;
	};

	// Defined by the two implementation files, and meant only for Vulkan_decoder::create() to call.
	// Declared here rather than in a private header so the three translation units cannot disagree
	// about their signature.
	Vulkan_decoder* make_decoder_chain_rebuilt(int device_id);
	Vulkan_decoder* make_decoder_chain_cached (int device_id);
}
