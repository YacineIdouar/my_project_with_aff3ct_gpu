/**
 * Channel_AWGN_LLR_hiprand_batched_hip.hip
 * ----------------------------------------
 * hipRAND AWGN channel of Channel_AWGN_LLR_hiprand_batched (HIP port of Channel_AWGN_LLR_curand_batched_cuda.cu): one
 * thread per pair of samples, hiprand_normal2() for the Box-Muller pair, the state written back so that the next call
 * continues its sequence. The module hands it all the frames of a firing at once, as one array.
 *
 * Only the device API of hipRAND is used (hiprand_kernel.h). It is header-only on both HIP platforms (rocRAND on AMD,
 * cuRAND on NVIDIA), so nothing has to be linked.
 */

#include <cstdint>
#include <sstream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include "Device/Hip/Hip_dispatch.hpp"
#include "Hip/Channel_AWGN_LLR_hiprand_batched_hip.hpp"

namespace
{
constexpr unsigned THREADS = 256;

void
check(const hipError_t error, const char* what)
{
    if (error == hipSuccess) return;
    std::stringstream message;
    message << what << ": " << hipGetErrorString(error) << ".";
    throw spu::tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
}

// one state per pair of samples; 'subsequence = idx' makes the states statistically independent
__global__ void
init_states_kernel(hiprandStateXORWOW_t* __restrict__ states, const uint32_t n_states, const unsigned long long seed)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) hiprand_init(seed, (unsigned long long)idx, 0ull, &states[idx]);
}

// y = x + sigma * n, n ~ N(0,1), two samples per thread
__global__ void
awgn_add_noise_kernel(const float* __restrict__ x,
                      float* __restrict__ y,
                      const uint32_t n,
                      const float sigma,
                      hiprandStateXORWOW_t* __restrict__ states)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t base = idx * 2;
    if (base >= n) return;

    hiprandStateXORWOW_t local_state = states[idx];
    const float2 noise = hiprand_normal2(&local_state);
    y[base] = x[base] + sigma * noise.x;
    if (base + 1 < n) y[base + 1] = x[base + 1] + sigma * noise.y;
    states[idx] = local_state;
}
}

sp_hip::Hip_channel_batched::Hip_channel_batched(const int dev_id, const size_t max_samples, const uint64_t seed)
  : dev_id(dev_id)
{
    this->init(max_samples, seed);
}

sp_hip::Hip_channel_batched::~Hip_channel_batched()
{
    if (this->d_states != nullptr)
    {
        // nothing to report from a destructor: the return codes are dropped on purpose
        (void)hipSetDevice(this->dev_id);
        (void)hipFree(this->d_states);
    }
}

void
sp_hip::Hip_channel_batched::init(const size_t max_samples, const uint64_t seed)
{
    check(hipSetDevice(this->dev_id), "hipSetDevice");
    const size_t n_states = (max_samples + 1) / 2;
    if (n_states != this->n_states)
    {
        if (this->d_states != nullptr) check(hipFree(this->d_states), "hipFree (hipRAND states)");
        this->d_states = nullptr;
        check(hipMalloc(&this->d_states, n_states * sizeof(hiprandStateXORWOW_t)), "hipMalloc (hipRAND states)");
        this->n_states = n_states;
    }

    // Setup, not part of a firing: launched directly and waited for -- on a non-blocking stream of its own, not on the
    // null stream with a hipDeviceSynchronize(). While a stream capture is in flight anywhere in the process (another
    // GPU task recording its HIP graph), the HIP runtime fails a device synchronisation or a null-stream operation and
    // invalidates every capture in flight, whatever its mode.
    hipStream_t setup = nullptr;
    check(hipStreamCreateWithFlags(&setup, hipStreamNonBlocking), "hipStreamCreateWithFlags (hipRAND states)");
    const unsigned blocks = (unsigned)((n_states + THREADS - 1) / THREADS);
    init_states_kernel<<<blocks, THREADS, 0, setup>>>(
      static_cast<hiprandStateXORWOW_t*>(this->d_states), (uint32_t)n_states, (unsigned long long)seed);
    const hipError_t launch_error = hipGetLastError();
    const hipError_t sync_error = launch_error == hipSuccess ? hipStreamSynchronize(setup) : hipSuccess;
    (void)hipStreamDestroy(setup);
    check(launch_error, "init_states_kernel launch");
    check(sync_error, "init_states_kernel");
}

std::vector<spu::device_interface::launch_timing>
sp_hip::Hip_channel_batched::add_noise(const float* x,
                                       float* y,
                                       const size_t n,
                                       const float sigma,
                                       spu::device_interface::GpuStream* stream)
{
    const uint32_t pairs = (uint32_t)((n + 1) / 2);
    const unsigned blocks = (pairs + THREADS - 1) / THREADS;
    return spu::sp_hip::dispatch(stream,
                                 awgn_add_noise_kernel,
                                 blocks,
                                 THREADS,
                                 0,
                                 x,
                                 y,
                                 (uint32_t)n,
                                 sigma,
                                 static_cast<hiprandStateXORWOW_t*>(this->d_states));
}
