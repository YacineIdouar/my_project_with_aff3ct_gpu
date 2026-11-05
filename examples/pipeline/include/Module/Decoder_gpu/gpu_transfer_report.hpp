#pragma once

/*
The copies made by StreamPU's adaptors between a CPU stage and a GPU stage.

With SPU_GPU_TRANSFER=copy (or auto, the default, on a discrete GPU), each adaptor slot between a CPU
stage and a GPU stage holds a couple <pinned host buffer; device buffer>: the CPU stage sees the host
half, the GPU stage the device half, and the adaptor copies between the two halves on a copy stream
of the GPU. With SPU_GPU_TRANSFER=swap (or auto on an integrated GPU), the two stages share one
buffer and nothing is copied. See roadmap/discrete_gpu_transfers.md in StreamPU.

StreamPU counts the copies in spu::device_interface::gpu_transfer_stats(), one set per direction.
print_report() shows them per copy:
  - GPU   = device time of one copy. Measured only with the GPU timing of its API on
            (SPU_CUDA_PROFILE_GPU, SPU_HIP_PROFILE_GPU, SPU_SYCL_PROFILE_GPU or
            SPU_VULKAN_PROFILE_GPU set to 1), n/a otherwise;
  - QUEUE = CPU time to queue one copy (on Vulkan, it includes waiting for the previous copy of
            the copy stream);
  - WAIT  = CPU time the stages spent waiting for the copies, spread over the copies. A copy first
            waits for the GPU stage to be done with the buffer it overwrites, so WAIT also grows
            when the GPU stage is the bottleneck: compare it with GPU.

The counters cover every exec() of the process, and are complete once pipeline->exec() has
returned (the device times of the last copies are read when the pipeline resets its adaptors).

A StreamPU without these counters (older than the copies at the adaptors) still builds:
CMakeLists.txt defines HAS_SPU_TRANSFER_STATS only when the installed StreamPU has them.
*/

#include <cstdio>
#include <ostream>
#include <string>

#include <streampu.hpp>

namespace gpu_transfer
{

// The transfer mode at the adaptors, for the run header.
inline std::string describe()
{
#ifdef HAS_SPU_TRANSFER_STATS
	return std::string(spu::device_interface::gpu_transfer_to_string(spu::device_interface::gpu_transfer_mode())) +
	       " (SPU_GPU_TRANSFER)";
#else
	return "swap (this StreamPU has no copies at the adaptors)";
#endif
}

// The copies, per direction and per copy. Prints nothing when there was none (the stages swapped
// shared buffers). To be called after pipeline->exec() has returned.
inline void print_report(std::ostream& os)
{
#ifdef HAS_SPU_TRANSFER_STATS
	namespace di = spu::device_interface;
	const di::gpu_transfer_counters& stats = di::gpu_transfer_stats();
	if (stats.h2d.copies.load() == 0 && stats.d2h.copies.load() == 0) return;

	const char* sep = "# ----------|----------|----------||----------|----------||----------|----------|----------";

	os << "#" << std::endl;
	os << "# Copies at the adaptors (couples <host; device>)" << std::endl;
	os << "#   GPU   = device time of one copy (n/a: the GPU timing of its API is off)" << std::endl;
	os << "#   QUEUE = CPU time to queue one copy" << std::endl;
	os << "#   WAIT  = CPU time the stages waited for the copies, per copy (back-pressure of the GPU stage"
	   << std::endl;
	os << "#           included: compare it with GPU)" << std::endl;
	os << sep << std::endl;
	os << "#  DIRECTION |   COPIES |  KB/COPY ||  GPU AVG |      GPU ||  QUEUE AV |  WAIT AVG |  CPU WAITS"
	   << std::endl;
	os << "#            |          |          ||     (us) |   (GB/s) ||      (us) |      (us) |           "
	   << std::endl;
	os << sep << std::endl;

	auto row = [&os](const char* name, const di::gpu_transfer_direction_counters& c)
	{
		const unsigned long long copies = c.copies.load();
		if (copies == 0) return;
		const double n = (double)copies;
		const double bytes = (double)c.bytes.load();
		const unsigned long long gpu_timed = c.gpu_timed.load();

		char head[128];
		std::snprintf(head, sizeof(head), "# %10s | %8llu | %8.1f ", name, copies, bytes / n / 1024.0);

		char gpu[128];
		if (gpu_timed > 0)
		{
			const double gpu_ns = (double)c.gpu_ns.load();
			const double avg_bytes = bytes / n; // every copy of a direction moves the same size, per socket
			std::snprintf(gpu,
			              sizeof(gpu),
			              "|| %8.2f | %8.2f ",
			              gpu_ns / (double)gpu_timed / 1000.0,
			              gpu_ns > 0 ? avg_bytes * (double)gpu_timed / gpu_ns : 0.0);
		}
		else
			std::snprintf(gpu, sizeof(gpu), "|| %8s | %8s ", "n/a", "n/a");

		char cpu[128];
		std::snprintf(cpu,
		              sizeof(cpu),
		              "|| %9.2f | %9.2f | %10llu",
		              (double)c.issue_ns.load() / n / 1000.0,
		              (double)c.wait_ns.load() / n / 1000.0,
		              (unsigned long long)c.waits.load());

		os << head << gpu << cpu << std::endl;
	};
	row("CPU -> GPU", stats.h2d);
	row("GPU -> CPU", stats.d2h);
	os << sep << std::endl;
#else
	(void)os;
#endif
}

} // namespace gpu_transfer
