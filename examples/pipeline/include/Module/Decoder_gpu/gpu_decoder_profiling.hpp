#pragma once

/*
Profiling shared by every GPU module of this example -- the LDPC decoders and the AWGN channels --
across the four backends (CUDA, HIP, SYCL, Vulkan).

Every backend's executor already knows how to time a launch -- CUDA_executor and HIP_executor with
a pair of events around the kernel, VULKAN_executor with a pair of GPU timestamps around the
dispatch chain -- and each hands the result back as a vector of (cpu_us, gpu_us) pairs from
return_profiling_times(). What none of them does is survive the decode: the records live on the
executor, and reporting them means someone has to collect and aggregate them.

That is all this header is. One accumulator per handler instance, registered in a process-wide list
so main() can print a report without holding a pointer to objects that the modules (and their
clones) built themselves.

Recording takes no lock, and must not need one: every handler belongs to exactly one replica of its
module, and every replica to one thread. That holds because each module's clone() builds fresh
handlers rather than copying the pointer -- for the channels as much as for the decoders. Anything
that goes back to sharing a handler between threads has to add its own synchronisation here first.

Everything is inline: the CUDA file goes through nvcc, the SYCL one through a SYCL compiler and the
rest through the host compiler, and a header-only facility spares the build from having to place a
single .cpp in exactly one of those object libraries. The function-local statics below are still a
single instance across all of them, which is what the registry needs.

One backend does not report device time. A SYCL queue can only be profiled if it was created with
sycl::property::queue::enable_profiling up front, and the queue a task hands the decoder was not
(see SyclStream's constructor in StreamPU); SYCL_executor works around it by keeping a *separate*
profiling queue, which would mean running the work somewhere other than where it normally runs.
Rather than measure a different execution than the one under test, the SYCL modules report their
host-side cost and leave the device column empty. The report prints 'n/a' for them.
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

// Totals for one handler instance. 'calls' counts the module-level operations (one ldpc_decode(),
// one add_noise()); 'launches' counts the individual kernel launches folded into them -- 2 + 2*n_ite
// per decode on CUDA/HIP/SYCL, exactly 1 on Vulkan (the whole chain is one submit), and 1 either way
// for the channel.
//
// The min/max are over calls, not over launches: a call is the unit that matters, and it is the only
// unit the backends agree on.
struct stats
{
	uint64_t calls      = 0;
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
	std::string module;  // "DECODER" / "CHANNEL"
	std::string backend; // "CUDA" / "HIP" / "SYCL" / "VULKAN"
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
// first frame: a handler reads it on every call, and a run that flips it halfway produces a report
// whose averages cover only part of the frames.
inline bool enabled() { return detail::enabled_storage(); }
inline void set_enabled(bool on) { detail::enabled_storage() = on; }

// One per handler instance, held by value as a member. Writes come only from the thread that owns
// the handler (see the note at the top of this file), so recording needs no lock; only registration
// and the final report take one.
class accumulator
{
  public:
	accumulator(const char* module, const char* backend, bool gpu_timed = true)
	{
		s.gpu_timed = gpu_timed;
		std::lock_guard<std::mutex> lock(detail::registry_mutex());
		detail::registry().push_back(detail::entry{ module, backend, &s });
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

	// One whole call, from a backend that timed each of its launches separately: the launches are
	// summed into a single call-level (cpu, gpu) pair. Summing rather than averaging is the point --
	// what a frame costs is the sum of what its kernels cost, and that is what compares against
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

	// One whole call, already summed.
	void add(float cpu_us, float gpu_us, uint64_t launches = 1)
	{
		if (s.calls == 0)
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
		s.calls++;
	}

	const stats& get() const { return s; }

  private:
	stats s;
};

// One row per handler that recorded something, plus a total, and nothing at all when none did.
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
	bool all_gpu_timed = true;
	for (const auto& e : snapshot)
	{
		const stats& s = *e.s;
		if (s.calls == 0) continue;

		if (total.calls == 0)
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
		total.calls += s.calls;
		// One untimed backend in the mix makes the device total meaningless, so the whole column
		// goes to 'n/a' rather than silently reporting a partial sum as if it were complete.
		all_gpu_timed = all_gpu_timed && s.gpu_timed;
	}

	if (total.calls == 0) return;

	const char* sep = "# ---------|---------|----|----------|----------"
	                  "||----------|----------|----------||----------|----------|----------";

	os << "#" << std::endl;
	os << "# GPU module profiling" << std::endl;
	os << "#   CPU = host-side cost of one call (enqueue / submit), device wait excluded" << std::endl;
	os << "#   GPU = device time of that call, summed over the kernels it launched" << std::endl;
	if (!all_gpu_timed)
		os << "#   n/a = this backend's task queue was not created with profiling enabled" << std::endl;
	os << sep << std::endl;
	os << "#   MODULE | BACKEND | ID |    CALLS |  KERNELS "
	      "||  CPU AVG |  CPU MIN |  CPU MAX ||  GPU AVG |  GPU MIN |  GPU MAX " << std::endl;
	os << "#          |         |    |          |          "
	      "||     (us) |     (us) |     (us) ||     (us) |     (us) |     (us) " << std::endl;
	os << sep << std::endl;

	auto row = [&os](const std::string& module,
	                 const std::string& backend,
	                 const std::string& id,
	                 const stats& s)
	{
		char head[128];
		std::snprintf(head,
		              sizeof(head),
		              "# %8s | %7s | %2s | %8llu | %8llu ",
		              module.c_str(),
		              backend.c_str(),
		              id.c_str(),
		              (unsigned long long)s.calls,
		              (unsigned long long)s.launches);

		char cpu[128];
		std::snprintf(cpu,
		              sizeof(cpu),
		              "|| %8.2f | %8.2f | %8.2f ",
		              s.cpu_us_sum / (double)s.calls,
		              (double)s.cpu_us_min,
		              (double)s.cpu_us_max);

		char gpu[128];
		if (s.gpu_timed)
			std::snprintf(gpu,
			              sizeof(gpu),
			              "|| %8.2f | %8.2f | %8.2f",
			              s.gpu_us_sum / (double)s.calls,
			              (double)s.gpu_us_min,
			              (double)s.gpu_us_max);
		else
			std::snprintf(gpu, sizeof(gpu), "|| %8s | %8s | %8s", "n/a", "n/a", "n/a");

		os << head << cpu << gpu << std::endl;
	};

	// Grouped by module then backend, so replicated handlers of the same thing sit together and the
	// per-group subtotal is the number worth reading when a stage runs on several threads.
	size_t id = 0;
	for (const auto& e : snapshot)
	{
		if (e.s->calls == 0) continue;
		row(e.module, e.backend, std::to_string(id++), *e.s);
	}

	os << sep << std::endl;
	total.gpu_timed = all_gpu_timed;
	row("TOTAL", "*", "*", total);
	os << "#" << std::endl;
}

}
