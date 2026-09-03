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
strategies is the whole point of having them, so the choice belongs on the command line rather
than in the source: --gpu-dispatch in main_gpu.cpp, or SPU_GPU_DISPATCH_MODE in the environment.

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
inline mode& storage()
{
	static mode m = []()
	{
		// CACHED unless the environment explicitly asks otherwise, matching what both StreamPU
		// backends default to.
		const char* env = std::getenv("SPU_GPU_DISPATCH_MODE");
		if (env == nullptr) return mode::CACHED;

		const std::string s(env);
		if (s == "one_shot" || s == "ONE_SHOT" || s == "0") return mode::ONE_SHOT;
		return mode::CACHED;
	}();
	return m;
}
}

inline mode get() { return detail::storage(); }

// Set it before the first frame -- and, more strictly, before any module builds an executor:
// StreamPU reads these variables once, on the first launch, and caches the answer.
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

}
