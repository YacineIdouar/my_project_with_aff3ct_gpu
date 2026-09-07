#pragma once

/*
One switch, both backends: whether a GPU module replays a *recorded* chain of kernels or issues
them one by one.

StreamPU offers the same choice twice, under two names and two mechanisms:

  Vulkan  -- vk_dispatch_mode::{CACHED, ONE_SHOT}. CACHED records the chain into a VkCommandBuffer
             once per set of dispatch parameters and resubmits it verbatim; ONE_SHOT re-records
             the stream's scratch buffer every frame.
  CUDA    -- cuda_dispatch_mode::{CACHED, ONE_SHOT}. CACHED captures the chain into a CUDA graph
             once and relaunches it with a single cudaGraphLaunch(); ONE_SHOT issues the kernels
             with cudaLaunchKernel(), as before graphs existed.

Both replace N per-frame launches with one submit, both are keyed on the launch parameters, and
both fall back to the plain path when the caller varies those parameters. Comparing the two
strategies is the whole point of having them, so the choice lives in the environment and nowhere
else: SPU_GPU_DISPATCH_MODE for both backends at once, or StreamPU's own
SPU_{CUDA,VULKAN,HIP}_DISPATCH_MODE for one of them. There is deliberately no command-line switch:
a binary-level flag would be a second way to say the same thing, and the two could disagree.

Setting a per-backend variable on its own works and is reported as such: the mode this header
reports is derived from whichever variables are set, never assumed, so the run header can never say
"cached" while StreamPU one-shots.

This works by setting StreamPU's own SPU_CUDA_DISPATCH_MODE / SPU_VULKAN_DISPATCH_MODE, which is
now the only way to choose: StreamPU dropped the per-executor set_dispatch_mode() when it collapsed
to a single launch path, on the grounds that a no-graph run is a whole-run comparison rather than a
per-module setting. Those variables are read once, on the first launch, so set() has to run before
any module builds an executor -- which is where main_gpu.cpp calls it, from argument parsing.

Consequence worth knowing: the choice is now process-wide rather than per executor. For this example
that is the same thing, since it owns every executor in its process.
*/

#include <cstdlib>
#include <string>

namespace gpu_dispatch
{

enum class mode
{
	CACHED,  // record/capture once, replay afterwards
	ONE_SHOT // rebuild and reissue every frame
};

namespace detail
{
// The three variables StreamPU actually reads, in the order describe() reports them.
inline const char* const* backend_vars()
{
	static const char* const names[] = { "SPU_CUDA_DISPATCH_MODE",
	                                     "SPU_VULKAN_DISPATCH_MODE",
	                                     "SPU_HIP_DISPATCH_MODE" };
	return names;
}

inline const char* const* backend_names()
{
	static const char* const names[] = { "cuda", "vulkan", "hip" };
	return names;
}

// Byte for byte what VULKAN_executor::default_dispatch_mode() and its CUDA/HIP twins do, so that
// what this example reports and what StreamPU does can never diverge on a value one of them accepts
// and the other does not.
inline mode parse_env(const char* value)
{
	if (value == nullptr) return mode::CACHED;
	const std::string s(value);
	if (s == "one_shot" || s == "ONE_SHOT" || s == "0") return mode::ONE_SHOT;
	return mode::CACHED;
}

inline mode& storage()
{
	static mode m = []()
	{
		const char* aggregate = std::getenv("SPU_GPU_DISPATCH_MODE");

		if (aggregate != nullptr)
		{
			const mode chosen = parse_env(aggregate);

			// Forward to StreamPU's own per-backend variables here rather than in set(): with the
			// command-line switch gone, reading the aggregate is the only moment the choice is
			// made, so this is what makes SPU_GPU_DISPATCH_MODE reach StreamPU at all. Only ever
			// fills in what the user has not set directly -- the 0 in setenv() means "do not
			// overwrite" -- so setting SPU_VULKAN_DISPATCH_MODE alone still wins for that backend.
			const char* value = (chosen == mode::CACHED) ? "cached" : "one_shot";
			for (int i = 0; i < 3; ++i) setenv(backend_vars()[i], value, 0);
			return chosen;
		}

		// No aggregate: the mode is whatever the per-backend variables say, because those are what
		// StreamPU will read. Deriving it rather than defaulting to CACHED is the whole point --
		// reporting "cached" while StreamPU one-shots because SPU_VULKAN_DISPATCH_MODE was set
		// directly is a banner that lies about the run it is describing.
		//
		// When they disagree no single value is right; CACHED is returned as the neutral answer and
		// describe() spells the split out instead.
		mode agreed = mode::CACHED;
		bool seen = false;
		for (int i = 0; i < 3; ++i)
		{
			const char* v = std::getenv(backend_vars()[i]);
			if (v == nullptr) continue;

			const mode here = parse_env(v);
			if (!seen)
			{
				agreed = here;
				seen = true;
			}
			else if (here != agreed)
				return mode::CACHED;
		}
		return agreed;
	}();
	return m;
}
}

inline mode get() { return detail::storage(); }

// Kept for programmatic use; nothing in this example calls it any more, since the mode comes from
// the environment. Set it before the first frame -- and, more strictly, before any module builds an
// executor: StreamPU reads these variables once, on the first launch, and caches the answer.
inline void set(mode m)
{
	detail::storage() = m;
	const char* value = (m == mode::CACHED) ? "cached" : "one_shot";
	setenv("SPU_CUDA_DISPATCH_MODE", value, 1);
	setenv("SPU_VULKAN_DISPATCH_MODE", value, 1);
	setenv("SPU_HIP_DISPATCH_MODE", value, 1);
}

inline bool parse(const std::string& s, mode& out)
{
	if (s == "cached" || s == "CACHED" || s == "1")
	{
		out = mode::CACHED;
		return true;
	}
	if (s == "one_shot" || s == "ONE_SHOT" || s == "oneshot" || s == "0")
	{
		out = mode::ONE_SHOT;
		return true;
	}
	return false;
}

inline const char* to_str(mode m) { return m == mode::CACHED ? "cached" : "one_shot"; }

// What to print in a run header: the single word when every backend ends up on the same mode, and
// the per-backend breakdown when they do not (which only happens if the SPU_*_DISPATCH_MODE
// variables were set individually and disagree). Resolving get() first matters: that is what
// forwards an aggregate setting into the per-backend variables this then reads back.
inline std::string describe()
{
	const mode aggregate = get();

	mode first = aggregate;
	bool uniform = true;
	for (int i = 0; i < 3; ++i)
		if (detail::parse_env(std::getenv(detail::backend_vars()[i])) != first) uniform = false;

	if (uniform) return to_str(aggregate);

	std::string out;
	for (int i = 0; i < 3; ++i)
	{
		if (!out.empty()) out += " ";
		out += detail::backend_names()[i];
		out += "=";
		out += to_str(detail::parse_env(std::getenv(detail::backend_vars()[i])));
	}
	return out;
}

}
