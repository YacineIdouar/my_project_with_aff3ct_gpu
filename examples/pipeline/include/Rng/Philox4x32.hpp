#ifndef PHILOX_4X32_HPP_
#define PHILOX_4X32_HPP_

/**
 * Philox4x32-10 + Box-Muller — backend-agnostic AWGN noise generator.
 * ===================================================================
 *
 * Portability contract
 * --------------------
 * This header deliberately depends on NOTHING but the C++ language and <cmath>:
 *
 *   - no cuRAND / rocRAND / oneMKL / Random123, no library of any kind;
 *   - no vendor intrinsics (no __umulhi, no __uint2float_rn, no sincosf);
 *   - no vendor vector types (no uint4, no float2) — plain arrays only;
 *   - no 128-bit or vendor-specific integer types, only uint32_t / uint64_t.
 *
 * The one thing it needs from the language is a 32x32 -> 64 bit unsigned multiply, which
 * CUDA, HIP and SYCL all get from `uint64_t` (nvcc/hipcc emit a single mul.hi.u32 for the
 * high half, so this costs nothing versus __umulhi). GLSL has no 64-bit integers, so the
 * Vulkan port in src/vulkan/Shaders/philox4x32.glsl uses the umulExtended() built-in for
 * that single line and is otherwise a literal transliteration of the code below.
 *
 * A backend kernel therefore reduces to: compute a global index, call philox_awgn_noise4()
 * once, add the four samples to four inputs. See awgn_add_noise_philox in
 * src/cuda/Channel_AWGN_LLR_prng_cuda.cu for the reference kernel.
 *
 * Why counter-based
 * -----------------
 * Philox is a *counter-based* generator: rather than iterating a state it acts like a block
 * cipher, mapping (counter, key) -> 4 pseudo-random 32-bit words. Nothing is stored between
 * calls, which is what makes it portable and cheap here:
 *
 *   - no per-thread state buffer to allocate, initialise, or resize when the frame count
 *     changes — so no device-side setup step to reimplement in each API;
 *   - no read-modify-write of shared state, so cloned modules running concurrent kernels on
 *     different streams cannot race on the RNG;
 *   - the stream is a pure function of (seed, counter, index), so a run replays exactly and
 *     every backend produces bit-identical noise for the same inputs.
 *
 * Quality: Philox4x32-10 passes BigCrush; it is the same generator cuRAND exposes as
 * CURAND_RNG_PSEUDO_PHILOX4_32_10. Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random
 * Numbers: As Easy as 1, 2, 3", SC'11.
 */

#include <cmath>
#include <cstdint>

// CUDA and HIP need the callee marked as device code; SYCL and plain C++ do not.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define PHILOX_FN __host__ __device__ inline
#else
#define PHILOX_FN inline
#endif

// Multipliers and Weyl-sequence key bumps, from the reference implementation.
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u // golden ratio
#define PHILOX_W1 0xBB67AE85u // sqrt(3) - 1

#define PHILOX_2PI 6.2831853071795864769f

// 32x32 -> 64 multiply, split into its high and low halves. This is the only operation that
// has to be spelled differently in GLSL (umulExtended); everywhere else uint64_t is enough.
PHILOX_FN void
philox_mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo)
{
    const uint64_t product = (uint64_t)a * (uint64_t)b;
    hi = (uint32_t)(product >> 32);
    lo = (uint32_t)product;
}

// One Philox round: two multiplies, then a fixed permutation of the four words mixing in
// the round key.
PHILOX_FN void
philox_round(uint32_t c[4], const uint32_t k[2])
{
    uint32_t hi0, lo0, hi1, lo1;
    philox_mulhilo(PHILOX_M0, c[0], hi0, lo0);
    philox_mulhilo(PHILOX_M1, c[2], hi1, lo1);

    const uint32_t c1 = c[1], c3 = c[3];
    c[0] = hi1 ^ c1 ^ k[0];
    c[1] = lo1;
    c[2] = hi0 ^ c3 ^ k[1];
    c[3] = lo0;
}

/**
 * The generator itself: (counter, key) -> four pseudo-random 32-bit words.
 * Ten rounds, the key bumped by the Weyl constants between them (round 0 uses the raw key).
 */
PHILOX_FN void
philox4x32_10(const uint32_t ctr[4], const uint32_t key[2], uint32_t out[4])
{
    uint32_t c[4] = { ctr[0], ctr[1], ctr[2], ctr[3] };
    uint32_t k[2] = { key[0], key[1] };

    for (int r = 0; r < 10; r++)
    {
        philox_round(c, k);
        if (r < 9)
        {
            k[0] += PHILOX_W0;
            k[1] += PHILOX_W1;
        }
    }

    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
    out[3] = c[3];
}

/**
 * Maps a random 32-bit word to (0, 1]. The +2^-33 offset keeps the result strictly positive
 * so the logf() below never sees 0; this is how cuRAND's own uniform conversion works, and
 * it caps the Gaussian tail at ~6.66 sigma exactly like cuRAND does.
 */
PHILOX_FN float
philox_uniform(uint32_t x)
{
    return (float)x * 2.3283064365386963e-10f + 1.1641532182693481e-10f;
}

/**
 * Box-Muller: two uniforms -> two independent N(0,1) samples.
 *
 * logf/cosf/sinf are the accurate versions on purpose. The fast intrinsics (__logf,
 * __sincosf) are cheaper but degrade the distribution tails, which is exactly where the FEC
 * error floor lives — and they are CUDA-only, which this header is not.
 */
PHILOX_FN void
philox_box_muller(uint32_t x0, uint32_t x1, float& z0, float& z1)
{
    const float r     = sqrtf(-2.0f * logf(philox_uniform(x0)));
    const float theta = PHILOX_2PI * philox_uniform(x1);

    z0 = r * cosf(theta);
    z1 = r * sinf(theta);
}

/**
 * One Philox draw -> four N(0,1) samples. This is the whole per-thread body of an AWGN
 * kernel, and the only function a backend has to call.
 *
 * @param index    Global thread index; makes each thread's draw unique within a launch
 * @param ctr_lo   Low  32 bits of the per-launch counter
 * @param ctr_hi   High 32 bits of the per-launch counter
 * @param key_lo   Low  32 bits of the seed
 * @param key_hi   High 32 bits of the seed
 * @param noise    Out: four independent standard normal samples
 */
PHILOX_FN void
philox_awgn_noise4(uint32_t index,
                   uint32_t ctr_lo,
                   uint32_t ctr_hi,
                   uint32_t key_lo,
                   uint32_t key_hi,
                   float    noise[4])
{
    // (counter, index) is unique across the whole simulation, so no two threads — in this
    // launch or any other — ever consume the same Philox output.
    const uint32_t ctr[4] = { index, ctr_lo, ctr_hi, 0u };
    const uint32_t key[2] = { key_lo, key_hi };

    uint32_t r[4];
    philox4x32_10(ctr, key, r);

    philox_box_muller(r[0], r[1], noise[0], noise[1]);
    philox_box_muller(r[2], r[3], noise[2], noise[3]);
}

#endif /* PHILOX_4X32_HPP_ */
