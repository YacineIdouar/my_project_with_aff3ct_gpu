/*
GLSL port of ../../../include/Rng/Philox4x32.hpp.

Line-for-line transliteration of the C++ header: same constants, same round function, same
uniform conversion, same Box-Muller. Given identical (index, counter, seed) it produces
bit-identical noise to the CUDA/HIP/SYCL path.

The single difference is philox_mulhilo(): GLSL has no 64-bit integers, so it uses the
umulExtended() built-in (GLSL 4.00+) instead of a uint64_t multiply. Everything else is
plain 32-bit integer and float arithmetic, so no extension is required.

Usage: '#include "philox4x32.glsl"' from a .comp shader (needs GL_GOOGLE_include_directive
and glslc -I on this directory), or paste the body in directly.
*/

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u // golden ratio
#define PHILOX_W1 0xBB67AE85u // sqrt(3) - 1

#define PHILOX_2PI 6.2831853071795864769

// 32x32 -> 64 multiply, split into halves. The C++ header uses uint64_t here.
void
philox_mulhilo(uint a, uint b, out uint hi, out uint lo)
{
    umulExtended(a, b, hi, lo);
}

// One Philox round: two multiplies, then a fixed permutation of the four words mixing in
// the round key.
void
philox_round(inout uvec4 c, uvec2 k)
{
    uint hi0, lo0, hi1, lo1;
    philox_mulhilo(PHILOX_M0, c.x, hi0, lo0);
    philox_mulhilo(PHILOX_M1, c.z, hi1, lo1);

    uint c1 = c.y, c3 = c.w;
    c.x = hi1 ^ c1 ^ k.x;
    c.y = lo1;
    c.z = hi0 ^ c3 ^ k.y;
    c.w = lo0;
}

// The generator itself: (counter, key) -> four pseudo-random 32-bit words.
uvec4
philox4x32_10(uvec4 ctr, uvec2 key)
{
    uvec4 c = ctr;
    uvec2 k = key;

    for (int r = 0; r < 10; r++)
    {
        philox_round(c, k);
        if (r < 9)
        {
            k.x += PHILOX_W0;
            k.y += PHILOX_W1;
        }
    }
    return c;
}

// Maps a random 32-bit word to (0, 1]: strictly positive so log() never sees 0.
float
philox_uniform(uint x)
{
    return float(x) * 2.3283064365386963e-10 + 1.1641532182693481e-10;
}

// Box-Muller: two uniforms -> two independent N(0,1) samples.
vec2
philox_box_muller(uint x0, uint x1)
{
    float r = sqrt(-2.0 * log(philox_uniform(x0)));
    float theta = PHILOX_2PI * philox_uniform(x1);
    return vec2(r * cos(theta), r * sin(theta));
}

// One Philox draw -> four N(0,1) samples: the whole per-thread body of an AWGN kernel.
vec4
philox_awgn_noise4(uint index, uint ctr_lo, uint ctr_hi, uint key_lo, uint key_hi)
{
    uvec4 r = philox4x32_10(uvec4(index, ctr_lo, ctr_hi, 0u), uvec2(key_lo, key_hi));

    vec2 g0 = philox_box_muller(r.x, r.y);
    vec2 g1 = philox_box_muller(r.z, r.w);
    return vec4(g0.x, g0.y, g1.x, g1.y);
}
