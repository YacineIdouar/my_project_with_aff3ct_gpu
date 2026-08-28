#pragma once

/*
Profiling shared by the four GPU decoder backends (CUDA, HIP, SYCL, Vulkan).

Every backend's executor already knows how to time a launch -- CUDA_executor and HIP_executor with
a pair of events around the kernel, VULKAN_executor with a pair of GPU timestamps around the
dispatch chain -- and each hands the result back as a vector of (cpu_us, gpu_us) pairs from
return_profiling_times(). What none of them does is survive the decode: the records live on the
executor, and reporting them means someone has to collect and aggregate them.

That is all this header is. One accumulator per decoder instance, registered in a process-wide list
so main() can print a report without holding a pointer to decoders that the modules (and their
clones) built themselves.

Everything is inline: the CUDA file goes through nvcc, the SYCL one through a SYCL compiler and the
rest through the host compiler, and a header-only facility spares the build from having to place a
single .cpp in exactly one of those object libraries. The function-local statics below are still a
single instance across all of them, which is what the registry needs.

One backend does not report device time. A SYCL queue can only be profiled if it was created with
sycl::property::queue::enable_profiling up front, and the queue a task hands the decoder was not
(see SyclStream's constructor in StreamPU); SYCL_executor works around it by keeping a *separate*
profiling queue, which would mean running the decode somewhere other than where it normally runs.
Rather than measure a different execution than the one under test, the SYCL decoder reports its
host-side cost and leaves the device column empty. The report prints 'n/a' for it.
*/

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace gpu_prof
{

// Totals for one decoder instance. 'decodes' counts ldpc_decode() calls; 'launches' counts the
// individual kernel launches folded into them, which is 2 + 2*n_ite per decode on CUDA/HIP/SYCL
// (one launch per kernel) and exactly 1 on Vulkan (the whole chain is one submit).
//
// The per-decode min/max are over decodes, not over launches: a decode is the unit that matters,
// and it is the only unit the four backends agree on.
struct stats
{
	uint64_t decodes    = 0;
	uint64_t launches   = 0;
	double   cpu_us_sum = 0.0;
	double   gpu_us_sum = 0.0;
	float    cpu_us_min = 0.f;
	float    cpu_us_max = 0.f;
	float    gpu_us_min = 0.f;
	float    gpu_us_max = 0.f;
	bool     gpu_timed  = true; // false on a backend that cannot time the device side
};

namespace detail
{
struct entry
{
	std::string backend;
	const stats* s;
};

inline std::mutex& registry_mutex()
{
	static std::mutex m;
	return m;
}

inline std::vector<entry>& registry()
{
	static std::vector<entry> r;
	return r;
}

inline bool& enabled_storage()
{
	static bool enabled = []()
	{
		// Same "1 enables" rule as StreamPU's own SPU_VULKAN_PROFILE, which this does not replace:
		// that one profiles every VULKAN_executor dispatch in the process, this one profiles the
		// LDPC decoders on whichever backend they run.
		const char* env = std::getenv("SPU_LDPC_PROFILE");
		return env != nullptr && std::string(env) == "1";
	}();
	return enabled;
}
}

// Off unless SPU_LDPC_PROFILE=1 or main_gpu.cpp turns it on from --dec-profile. Set it before the
// first decode: a decoder reads it on every call, but a run that flips it halfway produces a report
// whose averages cover only part of the frames.
inline bool enabled() { return detail::enabled_storage(); }
inline void set_enabled(bool on) { detail::enabled_storage() = on; }

// One per decoder instance, held by value as a member. Writes happen only from the thread that owns
// the decoder, so recording needs no lock; only registration and the final report take one.
class accumulator
{
  public:
	explicit accumulator(const char* backend, bool gpu_timed = true)
	{
		s.gpu_timed = gpu_timed;
		std::lock_guard<std::mutex> lock(detail::registry_mutex());
		detail::registry().push_back(detail::entry{ backend, &s });
	}

	~accumulator()
	{
		std::lock_guard<std::mutex> lock(detail::registry_mutex());
		auto& r = detail::registry();
		for (size_t i = 0; i < r.size(); ++i)
			if (r[i].s == &s)
			{
				r.erase(r.begin() + i);
				break;
			}
	}

	accumulator(const accumulator&) = delete;
	accumulator& operator=(const accumulator&) = delete;

	// One whole decode, from a backend that timed each of its launches separately: the launches are
	// summed into a single decode-level (cpu, gpu) pair. Summing rather than averaging is the point
	// -- what a frame costs is the sum of what its kernels cost, and that is what compares against
	// Vulkan's single chain submit.
	void add(const std::vector<std::pair<float, float>>& times)
	{
		if (times.empty()) return;

		float cpu_us = 0.f;
		float gpu_us = 0.f;
		for (const auto& t : times)
		{
			cpu_us += t.first;
			gpu_us += t.second;
		}
		add(cpu_us, gpu_us, times.size());
	}

	// One whole decode, already summed.
	void add(float cpu_us, float gpu_us, uint64_t launches = 1)
	{
		if (s.decodes == 0)
		{
			s.cpu_us_min = s.cpu_us_max = cpu_us;
			s.gpu_us_min = s.gpu_us_max = gpu_us;
		}
		else
		{
			s.cpu_us_min = std::min(s.cpu_us_min, cpu_us);
			s.cpu_us_max = std::max(s.cpu_us_max, cpu_us);
			s.gpu_us_min = std::min(s.gpu_us_min, gpu_us);
			s.gpu_us_max = std::max(s.gpu_us_max, gpu_us);
		}

		s.cpu_us_sum += cpu_us;
		s.gpu_us_sum += gpu_us;
		s.launches += launches;
		s.decodes++;
	}

	const stats& get() const { return s; }

  private:
	stats s;
};

// One row per decoder that recorded something, plus a total, and nothing at all when none did.
// Call it once the pipeline has stopped: it reads counters the decoding threads write without
// synchronisation.
inline void print_report(std::ostream& os)
{
	std::vector<detail::entry> snapshot;
	{
		std::lock_guard<std::mutex> lock(detail::registry_mutex());
		snapshot = detail::registry();
	}

	stats total;
	bool any_gpu_timed = false;
	for (const auto& e : snapshot)
	{
		const stats& s = *e.s;
		if (s.decodes == 0) continue;

		if (total.decodes == 0)
		{
			total.cpu_us_min = s.cpu_us_min;
			total.cpu_us_max = s.cpu_us_max;
			total.gpu_us_min = s.gpu_us_min;
			total.gpu_us_max = s.gpu_us_max;
		}
		else
		{
			total.cpu_us_min = std::min(total.cpu_us_min, s.cpu_us_min);
			total.cpu_us_max = std::max(total.cpu_us_max, s.cpu_us_max);
			total.gpu_us_min = std::min(total.gpu_us_min, s.gpu_us_min);
			total.gpu_us_max = std::max(total.gpu_us_max, s.gpu_us_max);
		}
		total.cpu_us_sum += s.cpu_us_sum;
		total.gpu_us_sum += s.gpu_us_sum;
		total.launches += s.launches;
		total.decodes += s.decodes;
		any_gpu_timed = any_gpu_timed || s.gpu_timed;
	}

	if (total.decodes == 0) return;

	const char* sep = "# --------|-----|----------|----------||----------|----------|----------"
	                  "||----------|----------|----------";

	os << "#" << std::endl;
	os << "# GPU LDPC decoder profiling" << std::endl;
	os << "#   CPU = host-side cost of launching one decode (enqueue / submit), device wait excluded"
	   << std::endl;
	os << "#   GPU = device time of that decode, summed over its kernels" << std::endl;
	if (!any_gpu_timed)
		os << "#   'n/a' = this backend's task queue was not created with profiling enabled" << std::endl;
	os << sep << std::endl;
	os << "# BACKEND | DEC |  DECODES |  KERNELS ||  CPU AVG |  CPU MIN |  CPU MAX "
	      "||  GPU AVG |  GPU MIN |  GPU MAX " << std::endl;
	os << "#         |     |          |          ||     (us) |     (us) |     (us) "
	      "||     (us) |     (us) |     (us) " << std::endl;
	os << sep << std::endl;

	auto row = [&os](const std::string& backend, const std::string& idx, const stats& s)
	{
		char buf[320];
		if (s.gpu_timed)
			std::snprintf(buf,
			              sizeof(buf),
			              "# %7s | %3s | %8llu | %8llu || %8.2f | %8.2f | %8.2f || %8.2f | %8.2f | %8.2f",
			              backend.c_str(),
			              idx.c_str(),
			              (unsigned long long)s.decodes,
			              (unsigned long long)s.launches,
			              s.cpu_us_sum / (double)s.decodes,
			              (double)s.cpu_us_min,
			              (double)s.cpu_us_max,
			              s.gpu_us_sum / (double)s.decodes,
			              (double)s.gpu_us_min,
			              (double)s.gpu_us_max);
		else
			std::snprintf(buf,
			              sizeof(buf),
			              "# %7s | %3s | %8llu | %8llu || %8.2f | %8.2f | %8.2f || %8s | %8s | %8s",
			              backend.c_str(),
			              idx.c_str(),
			              (unsigned long long)s.decodes,
			              (unsigned long long)s.launches,
			              s.cpu_us_sum / (double)s.decodes,
			              (double)s.cpu_us_min,
			              (double)s.cpu_us_max,
			              "n/a",
			              "n/a",
			              "n/a");
		os << buf << std::endl;
	};

	size_t idx = 0;
	std::string backends;
	for (const auto& e : snapshot)
	{
		if (e.s->decodes == 0) continue;
		if (backends.find(e.backend) == std::string::npos)
			backends += (backends.empty() ? "" : "+") + e.backend;
		row(e.backend, std::to_string(idx++), *e.s);
	}

	os << sep << std::endl;
	total.gpu_timed = any_gpu_timed;
	row(backends, "*", total);
	os << "#" << std::endl;
}

}
