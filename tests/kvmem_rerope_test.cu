// Unit test for the KVMem K2 re-phase kernel (standalone; build with
// `builder.exe -test tests/kvmem_rerope_test.cu`).
//
// Pipeline under test (int8-group64 K page, MRoPE text):
//   stored = Quant_g(H256(RoPE_src(raw)))
//   kernel re-bakes src -> dst; the result is compared against a fresh append at
//   the new slot, mirroring the exact float op sequence.
//
// The stored domain is the FULL-ROW normalized H256 rotation -- the same one the
// production append kernels, the attention Q path and the KVMem scorer use. It mixes
// all four 64-dim groups, so the re-phase has to rewrite every group: a per-group H64
// treatment of group 0 alone (what this kernel used to do) corrupts the moved page.
#include "kvmem/kvmem_rerope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::kvmem;

namespace {

constexpr int kHeads  = 4;
constexpr int kPages  = 2;
constexpr int kTokens = kPagedKVPageSize;  // 64
constexpr int kDim    = kKVCacheInt8HeadDim;
constexpr int kGroups = kKVCacheInt8Groups;

int failures = 0;
#define CHECK(cond, ...)                                      \
    do {                                                      \
        if (!(cond)) {                                        \
            ++failures;                                       \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__);   \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
        }                                                     \
    } while (0)

// Mirrors kTextRopeInvFrequency: theta^(-2*pair/64) with the Qwen3.6 text theta.
float host_freq(int pair) { return std::pow(1e7f, -2.0f * static_cast<float>(pair) / 64.0f); }

// -- warp-faithful H256 on a full 256-dim row: lane l owns dims l + 32*r --
// H32 across the 32 lanes, applied to each of the 8 register columns, then H8 across
// those columns, normalized by 2^-4. This is hadamard_d256.cuh's
// normalized_hadamard_d256_inplace as a host mirror.
void hadamard256_host(float* e) {
    for (int stride = 1; stride <= 16; stride <<= 1) {
        for (int r = 0; r < 8; ++r) {
            float tmp[32];
            for (int l = 0; l < 32; ++l) {
                const float v = e[l + 32 * r];
                const float p = e[(l ^ stride) + 32 * r];
                tmp[l]        = (l & stride) == 0 ? v + p : p - v;
            }
            for (int l = 0; l < 32; ++l) { e[l + 32 * r] = tmp[l]; }
        }
    }
    for (int span = 1; span < 8; span <<= 1) {
        for (int base = 0; base < 8; base += 2 * span) {
            for (int off = 0; off < span; ++off) {
                for (int l = 0; l < 32; ++l) {
                    const float lo = e[l + 32 * (base + off)];
                    const float hi = e[l + 32 * (base + off + span)];
                    e[l + 32 * (base + off)]        = lo + hi;
                    e[l + 32 * (base + off + span)] = lo - hi;
                }
            }
        }
    }
    for (int i = 0; i < 256; ++i) { e[i] *= 0x1p-4f; }
}

// MRoPE text: pair p in [0,32) pairs (p, p+32), axis = p % 3.
void rope_group0(float* e, const std::int32_t pos3[3], bool inverse) {
    for (int p = 0; p < 32; ++p) {
        const float phi = static_cast<float>(pos3[p % 3]) * host_freq(p);
        float s = std::sin(phi);
        const float c = std::cos(phi);
        if (inverse) { s = -s; }
        const float a = e[p];
        const float b = e[p + 32];
        e[p]      = a * c - b * s;
        e[p + 32] = b * c + a * s;
    }
}

std::int64_t code_off(int pg, int h, int t, int d) {
    return static_cast<std::int64_t>(kDim) * kTokens * (static_cast<std::int64_t>(h) + kHeads * pg) +
           static_cast<std::int64_t>(kDim) * t + d;
}
std::int64_t scale_off(int pg, int h, int t, int g) {
    return static_cast<std::int64_t>(kGroups) * kTokens *
               (static_cast<std::int64_t>(h) + kHeads * pg) +
           static_cast<std::int64_t>(kGroups) * t + g;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::srand(1234);

    const std::size_t code_count  = static_cast<std::size_t>(kDim) * kTokens * kHeads * kPages;
    const std::size_t scale_count = static_cast<std::size_t>(kGroups) * kTokens * kHeads * kPages;

    // -- random raw K + source/window positions --
    std::vector<float> raw(code_count);
    for (auto& v : raw) { v = static_cast<float>(std::rand() % 2001 - 1000) / 1000.0f; }
    std::vector<std::int32_t> src_pos3(3 * kTokens * kPages);
    std::vector<std::int32_t> dst_pos3(3 * kTokens * kPages);
    for (int pg = 0; pg < kPages; ++pg) {
        for (int t = 0; t < kTokens; ++t) {
            const std::int32_t src = pg * kTokens + t + 100000;  // absolute position
            const std::int32_t dst = pg * kTokens + t;           // compacted window slot
            for (int a = 0; a < 3; ++a) {
                src_pos3[a * (kTokens * kPages) + pg * kTokens + t] = src;
                dst_pos3[a * (kTokens * kPages) + pg * kTokens + t] = dst;
            }
        }
    }

    // -- stored = Quant_g(H256(RoPE_src(raw))), the production append pipeline on host --
    // The append kernels rotate the WHOLE 256-dim row and then quantize each 64-dim group
    // with its own absmax/127 scale; RoPE touches only the rotary dims [0,64).
    std::vector<std::int8_t> stored_codes(code_count);
    std::vector<unsigned short> stored_scales(scale_count);
    std::vector<float> row(256);
    for (int pg = 0; pg < kPages; ++pg) {
        for (int h = 0; h < kHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                const std::int32_t pos3[3] = {src_pos3[(pg * kTokens + t) * 3 + 0],
                                              src_pos3[(pg * kTokens + t) * 3 + 1],
                                              src_pos3[(pg * kTokens + t) * 3 + 2]};
                for (int d = 0; d < 256; ++d) { row[d] = raw[code_off(pg, h, t, d)]; }
                rope_group0(row.data(), pos3, false);
                hadamard256_host(row.data());
                for (int g = 0; g < kGroups; ++g) {
                    float absmax = 0.0f;
                    for (int i = 0; i < 64; ++i) {
                        absmax = std::max(absmax, std::fabs(row[g * 64 + i]));
                    }
                    const unsigned short bits =
                        __half_as_ushort(__float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
                    const float s   = __half2float(__ushort_as_half(bits));
                    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
                    for (int i = 0; i < 64; ++i) {
                        const int q = std::max(
                            -127, std::min(127, static_cast<int>(std::lrintf(row[g * 64 + i] * inv))));
                        stored_codes[code_off(pg, h, t, g * 64 + i)] = static_cast<std::int8_t>(q);
                    }
                    stored_scales[scale_off(pg, h, t, g)] = bits;
                }
            }
        }
    }

    // -- device run: re-bake src -> dst --
    std::int8_t* d_codes = nullptr;
    unsigned short* d_scales = nullptr;
    std::int32_t *d_src = nullptr, *d_dst = nullptr;
    cudaMalloc(&d_codes, code_count);
    cudaMalloc(&d_scales, scale_count * 2);
    cudaMalloc(&d_src, src_pos3.size() * 4);
    cudaMalloc(&d_dst, dst_pos3.size() * 4);
    cudaMemcpy(d_codes, stored_codes.data(), code_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, stored_scales.data(), scale_count * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src_pos3.data(), src_pos3.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst_pos3.data(), dst_pos3.size() * 4, cudaMemcpyHostToDevice);
    kvmem_rerope_int8_g64_pages<kHeads>(d_codes, reinterpret_cast<__half*>(d_scales), kPages,
                                        d_src, d_dst, 0);
    cudaDeviceSynchronize();

    std::vector<std::int8_t> out_codes(code_count);
    std::vector<unsigned short> out_scales(scale_count);
    cudaMemcpy(out_codes.data(), d_codes, code_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_scales.data(), d_scales, scale_count * 2, cudaMemcpyDeviceToHost);

    // -- the moved page must equal a fresh append at the NEW slot --
    // Dequant the full row with its per-group scales, undo H256, move the RoPE phases,
    // re-apply H256 and requantize: exactly what the append path would have written at
    // the dst position. The old per-group-H64 treatment of group 0 alone lands far
    // outside these tolerances, so this check is what pins the rotation domain.
    double worst = 0.0;
    int bad_codes = 0, scale_mismatch = 0;
    for (int pg = 0; pg < kPages; ++pg) {
        for (int h = 0; h < kHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                const std::int32_t pos3[3] = {src_pos3[(pg * kTokens + t) * 3 + 0],
                                              src_pos3[(pg * kTokens + t) * 3 + 1],
                                              src_pos3[(pg * kTokens + t) * 3 + 2]};
                const std::int32_t posd3[3] = {dst_pos3[(pg * kTokens + t) * 3 + 0],
                                               dst_pos3[(pg * kTokens + t) * 3 + 1],
                                               dst_pos3[(pg * kTokens + t) * 3 + 2]};
                for (int d = 0; d < 256; ++d) {
                    const float s = __half2float(__ushort_as_half(
                        stored_scales[static_cast<std::size_t>(scale_off(pg, h, t, d / 64))]));
                    row[d] = static_cast<float>(stored_codes[code_off(pg, h, t, d)]) * s;
                }
                hadamard256_host(row.data());
                rope_group0(row.data(), pos3, true);
                rope_group0(row.data(), posd3, false);
                hadamard256_host(row.data());
                for (int g = 0; g < kGroups; ++g) {
                    float absmax = 0.0f;
                    for (int i = 0; i < 64; ++i) {
                        absmax = std::max(absmax, std::fabs(row[g * 64 + i]));
                    }
                    const unsigned short ref_bits = __half_as_ushort(
                        __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
                    const float ref_scale = __half2float(__ushort_as_half(ref_bits));
                    const float got_scale = __half2float(__ushort_as_half(
                        out_scales[static_cast<std::size_t>(scale_off(pg, h, t, g))]));
                    // GPU sincosf vs CPU sin/cos differ in the last ulp, so the absmax can
                    // round to the neighbouring fp16. Value-level tolerance, not bits.
                    if (std::fabs(got_scale - ref_scale) > 1e-3f * ref_scale + 1e-7f) {
                        ++scale_mismatch;
                    }
                    for (int i = 0; i < 64; ++i) {
                        const float got = static_cast<float>(
                                              out_codes[code_off(pg, h, t, g * 64 + i)]) *
                                          got_scale;
                        const double d = std::fabs(static_cast<double>(got) - row[g * 64 + i]);
                        worst          = std::max(worst, d);
                        if (d > 1.01f * got_scale) { ++bad_codes; }
                    }
                }
            }
        }
    }
    CHECK(scale_mismatch == 0, "scales must match a fresh append at the new slot (%d differ)",
          scale_mismatch);
    CHECK(bad_codes == 0, "codes must land within one step of a fresh append (%d bad)", bad_codes);
    std::printf("worst dequant diff = %.6f\n", worst);

    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
