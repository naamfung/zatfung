// Standalone verification for the sm_75 MMA lowerings in ops/common/mma.cuh
// (build and run with `builder.exe -test tests/mma_emulation_test.cu`).
//
// Turing's HMMA units only expose m16n8k8 (fp16) and m8n8k16 (int8), so every wider Ampere
// shape is reconstructed from those, and bf16/tf32 -- which have no Turing tensor-core path at
// all -- fall back to warp-shuffle SIMT FMA. Both kinds of lowering have to consume the *same
// fragment registers* as the instruction they replace, and a wrong register pairing silently
// transposes the accumulation instead of failing.
//
// This test pins that down. NINFER_SM75 is defined before including the header, so the wrappers
// under test are the production lowerings rather than a copy of them; the device it then runs on
// is sm_86, which still has the instructions being replaced. Both sides therefore run the real
// PTX instructions sharing the same fragment layout, and an identity that holds here holds on
// Turing too. That is the strongest check available without a Turing part in the machine.
//
// One further case is asserted: the alternative pairing `(a0, a2, b0) + (a1, a3, b1)` -- the one
// the RTX 2080 Ti port ships for mma_f16 -- is checked to *disagree*, which is what gives the
// rest of the file its discriminating power.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// Selects the lowering, then re-declares nothing, so `mma_*` below is the sm_75 production
// lowering compiled for an sm_86 target.
#define NINFER_SM75 1
#undef NINFER_SM86
#undef NINFER_SM89
#include "ops/common/mma.cuh"

using namespace ninfer::ops;

namespace {

int failures = 0;
#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            ++failures;                                     \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
        }                                                   \
    } while (0)

constexpr int kWarps = 32; // one fragment per lane per warp
constexpr int kLanes = kWarps * 32;

struct Frag {
    unsigned a0, a1, a2, a3, b0, b1;
};

// ---------------------------------------------------------------------------------------------
// The instructions being replaced. Each is sm_80+; the host driver skips the run when the
// visible device cannot execute them (it can here: the machine is an sm_86 part, and the point
// of the test is precisely that the emulation is checked against real hardware).
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ bool reference_available() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return true;
#else
    return false;
#endif
}

__device__ __forceinline__ void m16n8k8_ref(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                            unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void m16n8k16_f16_ref(float& c0, float& c1, float& c2, float& c3,
                                                 const Frag& f) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
}

__device__ __forceinline__ void m16n8k16_f16acc_ref(unsigned& c0, unsigned& c1, const Frag& f) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
}

__device__ __forceinline__ void m16n8k32_s8_ref(int& c0, int& c1, int& c2, int& c3, const Frag& f) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
}

__device__ __forceinline__ void m16n8k16_bf16_ref(float& c0, float& c1, float& c2, float& c3,
                                                  const Frag& f) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
}

__device__ __forceinline__ void m16n8k8_tf32_ref(float& c0, float& c1, float& c2, float& c3,
                                                 const Frag& f) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(f.a0), "r"(f.a1), "r"(f.a2), "r"(f.a3), "r"(f.b0), "r"(f.b1));
}

// ---------------------------------------------------------------------------------------------
// Kernels: one fragment per lane, hardware and lowering side by side.
// ---------------------------------------------------------------------------------------------

__global__ void k_f16(const Frag* in, float4* hw, float4* em) {
    const int i = blockIdx.x * 32 + (threadIdx.x & 31);
    if (!reference_available()) {
        return;
    }
    const Frag f = in[i];
    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
    m16n8k16_f16_ref(h0, h1, h2, h3, f);
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    mma_f16(e0, e1, e2, e3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    hw[i] = make_float4(h0, h1, h2, h3);
    em[i] = make_float4(e0, e1, e2, e3);
}

// The RTX 2080 Ti port's pairing, asserted to be wrong below.
__global__ void k_f16_altpair(const Frag* in, float4* hw, float4* alt) {
    const int i = blockIdx.x * 32 + (threadIdx.x & 31);
    if (!reference_available()) {
        return;
    }
    const Frag f = in[i];
    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
    m16n8k16_f16_ref(h0, h1, h2, h3, f);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    m16n8k8_ref(a0, a1, a2, a3, f.a0, f.a2, f.b0);
    m16n8k8_ref(a0, a1, a2, a3, f.a1, f.a3, f.b1);
    hw[i]  = make_float4(h0, h1, h2, h3);
    alt[i] = make_float4(a0, a1, a2, a3);
}

__global__ void k_f16acc(const Frag* in, uint2* hw, uint2* em) {
    const int i = blockIdx.x * 32 + (threadIdx.x & 31);
    if (!reference_available()) {
        return;
    }
    const Frag f = in[i];
    unsigned h0 = 0u, h1 = 0u;
    m16n8k16_f16acc_ref(h0, h1, f);
    unsigned e0 = 0u, e1 = 0u;
    mma_f16_f16acc(e0, e1, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    hw[i] = make_uint2(h0, h1);
    em[i] = make_uint2(e0, e1);
}

__global__ void k_s8(const Frag* in, int4* hw, int4* em) {
    const int i = blockIdx.x * 32 + (threadIdx.x & 31);
    if (!reference_available()) {
        return;
    }
    const Frag f = in[i];
    int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
    m16n8k32_s8_ref(h0, h1, h2, h3, f);
    int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
    mma_s8(e0, e1, e2, e3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    hw[i] = make_int4(h0, h1, h2, h3);
    em[i] = make_int4(e0, e1, e2, e3);
}

__global__ void k_bf16(const Frag* in, float4* hw, float4* em) {
    const int i = blockIdx.x * 32 + (threadIdx.x & 31);
    if (!reference_available()) {
        return;
    }
    const Frag f = in[i];
    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
    m16n8k16_bf16_ref(h0, h1, h2, h3, f);
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    mma_bf16(e0, e1, e2, e3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    hw[i] = make_float4(h0, h1, h2, h3);
    em[i] = make_float4(e0, e1, e2, e3);
}

__global__ void k_tf32(const Frag* in, float4* hw, float4* em) {
    const int i = blockIdx.x * 32 + (threadIdx.x & 31);
    if (!reference_available()) {
        return;
    }
    const Frag f = in[i];
    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
    m16n8k8_tf32_ref(h0, h1, h2, h3, f);
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    mma_tf32_bits(e0, e1, e2, e3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    hw[i] = make_float4(h0, h1, h2, h3);
    em[i] = make_float4(e0, e1, e2, e3);
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

std::mt19937 rng(0x5eed75u);

float rand_unit() {
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    return d(rng);
}

// The fp16 accumulate case ends up with raw .f16x2 bits, and cuda_fp16's narrowing helpers are
// device-only, so decode by hand on the host.
float half_bits_to_float(unsigned bits) {
    const unsigned sign = (bits >> 15) & 0x1u;
    const unsigned exp  = (bits >> 10) & 0x1Fu;
    const unsigned man  = bits & 0x3FFu;
    float value         = 0.0f;
    if (exp == 0) {
        value = std::ldexp(static_cast<float>(man), -24); // subnormals, and zero
    } else if (exp == 31) {
        value = (man != 0) ? std::numeric_limits<float>::quiet_NaN()
                           : std::numeric_limits<float>::infinity();
    } else {
        value = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
    }
    return sign != 0 ? -value : value;
}

unsigned pack_f16x2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

unsigned pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&b);
}

// tf32 keeps 10 explicit mantissa bits; callers hand the tensor core values already rounded
// this way, and the shuffle lowering reads the register verbatim, so the test must too.
unsigned pack_tf32(float v) {
    unsigned bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits &= 0xFFFFE000u;
    return bits;
}

unsigned pack_s8x4(int lo0, int lo1, int lo2, int lo3) {
    const auto byte = [](int v) { return static_cast<unsigned>(v) & 0xFFu; };
    return byte(lo0) | (byte(lo1) << 8) | (byte(lo2) << 16) | (byte(lo3) << 24);
}

int rand_s8() {
    std::uniform_int_distribution<int> d(-8, 8);
    return d(rng);
}

std::vector<Frag> make_f16_frags() {
    std::vector<Frag> out(kLanes);
    for (Frag& f : out) {
        f.a0 = pack_f16x2(rand_unit(), rand_unit());
        f.a1 = pack_f16x2(rand_unit(), rand_unit());
        f.a2 = pack_f16x2(rand_unit(), rand_unit());
        f.a3 = pack_f16x2(rand_unit(), rand_unit());
        f.b0 = pack_f16x2(rand_unit(), rand_unit());
        f.b1 = pack_f16x2(rand_unit(), rand_unit());
    }
    return out;
}

std::vector<Frag> make_bf16_frags() {
    std::vector<Frag> out(kLanes);
    for (Frag& f : out) {
        f.a0 = pack_bf16x2(rand_unit(), rand_unit());
        f.a1 = pack_bf16x2(rand_unit(), rand_unit());
        f.a2 = pack_bf16x2(rand_unit(), rand_unit());
        f.a3 = pack_bf16x2(rand_unit(), rand_unit());
        f.b0 = pack_bf16x2(rand_unit(), rand_unit());
        f.b1 = pack_bf16x2(rand_unit(), rand_unit());
    }
    return out;
}

std::vector<Frag> make_tf32_frags() {
    std::vector<Frag> out(kLanes);
    for (Frag& f : out) {
        f.a0 = pack_tf32(rand_unit());
        f.a1 = pack_tf32(rand_unit());
        f.a2 = pack_tf32(rand_unit());
        f.a3 = pack_tf32(rand_unit());
        f.b0 = pack_tf32(rand_unit());
        f.b1 = pack_tf32(rand_unit());
    }
    return out;
}

std::vector<Frag> make_s8_frags() {
    std::vector<Frag> out(kLanes);
    for (Frag& f : out) {
        f.a0 = pack_s8x4(rand_s8(), rand_s8(), rand_s8(), rand_s8());
        f.a1 = pack_s8x4(rand_s8(), rand_s8(), rand_s8(), rand_s8());
        f.a2 = pack_s8x4(rand_s8(), rand_s8(), rand_s8(), rand_s8());
        f.a3 = pack_s8x4(rand_s8(), rand_s8(), rand_s8(), rand_s8());
        f.b0 = pack_s8x4(rand_s8(), rand_s8(), rand_s8(), rand_s8());
        f.b1 = pack_s8x4(rand_s8(), rand_s8(), rand_s8(), rand_s8());
    }
    return out;
}

Frag* upload(const std::vector<Frag>& host) {
    Frag* dev = nullptr;
    if (cudaMalloc(&dev, host.size() * sizeof(Frag)) != cudaSuccess) {
        std::printf("cudaMalloc failed\n");
        std::exit(2);
    }
    if (cudaMemcpy(dev, host.data(), host.size() * sizeof(Frag), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::printf("cudaMemcpy failed\n");
        std::exit(2);
    }
    return dev;
}

template <class T>
T* alloc_device(std::size_t count) {
    T* dev = nullptr;
    if (cudaMalloc(&dev, count * sizeof(T)) != cudaSuccess) {
        std::printf("cudaMalloc failed\n");
        std::exit(2);
    }
    return dev;
}

template <class T>
void download(const T* dev, std::vector<T>& host) {
    if (cudaMemcpy(host.data(), dev, host.size() * sizeof(T), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::printf("cudaMemcpy D2H failed\n");
        std::exit(2);
    }
}

int device_cc_major() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return 0;
    }
    return prop.major;
}

// Compare one 4-lane-group float result set. `tol` is relative with an absolute floor.
void compare_floats(const std::vector<float4>& hw, const std::vector<float4>& em, float tol,
                    const char* what, float& worst_out) {
    float worst = 0.0f;
    int bad     = 0;
    for (std::size_t i = 0; i < hw.size(); ++i) {
        const float h[4] = {hw[i].x, hw[i].y, hw[i].z, hw[i].w};
        const float e[4] = {em[i].x, em[i].y, em[i].z, em[i].w};
        for (int k = 0; k < 4; ++k) {
            const float d = std::fabs(h[k] - e[k]);
            worst         = std::max(worst, d);
            if (d > tol * (1.0f + std::fabs(h[k]))) {
                ++bad;
            }
        }
    }
    worst_out = worst;
    CHECK(bad == 0, "%s: %d of %zu lanes outside tolerance %.1e", what, bad, hw.size() * 4, tol);
}

} // namespace

int main() {
    if (device_cc_major() < 8) {
        std::printf("device is not sm_80+; nothing to compare against\n");
        std::printf("failures=0  VERDICT: SKIP\n");
        return 0;
    }

    // ---- fp16, f32 accumulate: m16n8k16 vs 2 x m16n8k8 ----
    {
        const std::vector<Frag> host = make_f16_frags();
        Frag* dev                    = upload(host);
        auto* hw                     = alloc_device<float4>(kLanes);
        auto* em                     = alloc_device<float4>(kLanes);
        k_f16<<<kWarps, 32>>>(dev, hw, em);
        std::vector<float4> hh(kLanes), he(kLanes);
        download(hw, hh);
        download(em, he);
        float worst = 0.0f;
        // Both sides are the same fp32 accumulation in a different association: exact to fp32
        // rounding, so only accumulation order separates them.
        compare_floats(hh, he, 1e-5f, "mma_f16 vs m16n8k16", worst);
        std::printf("mma_f16        worst |hw - emul| = %.3e\n", worst);
        cudaFree(dev);
        cudaFree(hw);
        cudaFree(em);
    }

    // ---- negative control: the (a0, a2, b0) + (a1, a3, b1) pairing must NOT reproduce it ----
    {
        const std::vector<Frag> host = make_f16_frags();
        Frag* dev                    = upload(host);
        auto* hw                     = alloc_device<float4>(kLanes);
        auto* alt                    = alloc_device<float4>(kLanes);
        k_f16_altpair<<<kWarps, 32>>>(dev, hw, alt);
        std::vector<float4> hh(kLanes), ha(kLanes);
        download(hw, hh);
        download(alt, ha);
        int differing = 0;
        for (std::size_t i = 0; i < hh.size(); ++i) {
            const float h[4] = {hh[i].x, hh[i].y, hh[i].z, hh[i].w};
            const float a[4] = {ha[i].x, ha[i].y, ha[i].z, ha[i].w};
            for (int k = 0; k < 4; ++k) {
                if (std::fabs(h[k] - a[k]) > 1e-5f * (1.0f + std::fabs(h[k]))) {
                    ++differing;
                }
            }
        }
        CHECK(differing > 0,
              "the alternative (a0,a2,b0)+(a1,a3,b1) pairing was expected to disagree, but it "
              "matched everywhere -- the test has no discriminating power");
        std::printf("alt pairing    disagreements  = %d of %zu\n", differing, hh.size() * 4);
        cudaFree(dev);
        cudaFree(hw);
        cudaFree(alt);
    }

    // ---- fp16 accumulate: m16n8k16 f16-acc vs 2 x m16n8k8 f16-acc ----
    {
        const std::vector<Frag> host = make_f16_frags();
        Frag* dev                    = upload(host);
        auto* hw                     = alloc_device<uint2>(kLanes);
        auto* em                     = alloc_device<uint2>(kLanes);
        k_f16acc<<<kWarps, 32>>>(dev, hw, em);
        std::vector<uint2> hh(kLanes), he(kLanes);
        download(hw, hh);
        download(em, he);
        // Each partial sum rounds to fp16, where the hardware rounds once, so allow a few fp16
        // ulps rather than demanding bit equality.
        float worst = 0.0f;
        int bad     = 0;
        for (std::size_t i = 0; i < hh.size(); ++i) {
            const unsigned hr[2] = {hh[i].x, hh[i].y};
            const unsigned er[2] = {he[i].x, he[i].y};
            for (int r = 0; r < 2; ++r) {
                const float h[2] = {half_bits_to_float(hr[r] & 0xFFFFu),
                                    half_bits_to_float(hr[r] >> 16)};
                const float e[2] = {half_bits_to_float(er[r] & 0xFFFFu),
                                    half_bits_to_float(er[r] >> 16)};
                for (int k = 0; k < 2; ++k) {
                    const float d = std::fabs(h[k] - e[k]);
                    worst         = std::max(worst, d);
                    if (d > 2e-2f * (1.0f + std::fabs(h[k]))) {
                        ++bad;
                    }
                }
            }
        }
        CHECK(bad == 0, "mma_f16_f16acc: %d lanes outside tolerance", bad);
        std::printf("mma_f16_f16acc worst |hw - emul| = %.3e\n", worst);
        cudaFree(dev);
        cudaFree(hw);
        cudaFree(em);
    }

    // ---- int8: m16n8k32 vs 4 x m8n8k16 (exact integer arithmetic) ----
    {
        const std::vector<Frag> host = make_s8_frags();
        Frag* dev                    = upload(host);
        auto* hw                     = alloc_device<int4>(kLanes);
        auto* em                     = alloc_device<int4>(kLanes);
        k_s8<<<kWarps, 32>>>(dev, hw, em);
        std::vector<int4> hh(kLanes), he(kLanes);
        download(hw, hh);
        download(em, he);
        int bad = 0;
        for (std::size_t i = 0; i < hh.size(); ++i) {
            const int h[4] = {hh[i].x, hh[i].y, hh[i].z, hh[i].w};
            const int e[4] = {he[i].x, he[i].y, he[i].z, he[i].w};
            for (int k = 0; k < 4; ++k) {
                if (h[k] != e[k]) {
                    ++bad;
                }
            }
        }
        CHECK(bad == 0, "mma_s8 vs m16n8k32: %d elements differ (want exact)", bad);
        std::printf("mma_s8         exact mismatches = %d\n", bad);
        cudaFree(dev);
        cudaFree(hw);
        cudaFree(em);
    }

    // ---- bf16: m16n8k16 vs the warp-shuffle fp32 lowering ----
    {
        const std::vector<Frag> host = make_bf16_frags();
        Frag* dev                    = upload(host);
        auto* hw                     = alloc_device<float4>(kLanes);
        auto* em                     = alloc_device<float4>(kLanes);
        k_bf16<<<kWarps, 32>>>(dev, hw, em);
        std::vector<float4> hh(kLanes), he(kLanes);
        download(hw, hh);
        download(em, he);
        float worst = 0.0f;
        compare_floats(hh, he, 1e-5f, "mma_bf16 vs m16n8k16", worst);
        std::printf("mma_bf16       worst |hw - emul| = %.3e\n", worst);
        cudaFree(dev);
        cudaFree(hw);
        cudaFree(em);
    }

    // ---- tf32: m16n8k8 vs the warp-shuffle fp32 lowering ----
    {
        const std::vector<Frag> host = make_tf32_frags();
        Frag* dev                    = upload(host);
        auto* hw                     = alloc_device<float4>(kLanes);
        auto* em                     = alloc_device<float4>(kLanes);
        k_tf32<<<kWarps, 32>>>(dev, hw, em);
        std::vector<float4> hh(kLanes), he(kLanes);
        download(hw, hh);
        download(em, he);
        float worst = 0.0f;
        compare_floats(hh, he, 1e-5f, "mma_tf32_bits vs m16n8k8", worst);
        std::printf("mma_tf32_bits  worst |hw - emul| = %.3e\n", worst);
        cudaFree(dev);
        cudaFree(hw);
        cudaFree(em);
    }

    const bool ok = cudaDeviceSynchronize() == cudaSuccess;
    CHECK(ok, "kernel launch failed");
    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
