#pragma once

#include "core/arch_traits.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

#if defined(NINFER_SM75)
// =============================================================================================
// Turing (sm_75) tensor-core lowering.
//
// Turing's HMMA units expose exactly two shapes: m16n8k8 for fp16 and m8n8k16 for int8. Every
// wider Ampere shape below (m16n8k16, m16n8k32) is reconstructed from those two, and the two
// shapes Turing has no tensor-core path for at all -- bf16 and tf32 -- fall back to
// warp-shuffle SIMT FMA that consumes the *same* fragment registers, so no caller has to know
// which lowering it is compiling into.
//
// The register convention is the PTX one -- the very same registers the sm_86/89/120a asm paths
// below consume -- so each lowering is a pure re-association of the K dimension:
//
//   m16n8k16  A: a0 = row g   k 0-7    a1 = row g+8 k 0-7     B: b0 = B[k 0-7 ][n]
//                 a2 = row g   k 8-15   a3 = row g+8 k 8-15      b1 = B[k 8-15][n]
//   m16n8k32  A: a0 = row g   k 0-15   a1 = row g+8 k 0-15    B: b0 = B[k  0-15][n]
//                 a2 = row g   k 16-31  a3 = row g+8 k 16-31     b1 = B[k 16-31][n]
//
// Note the pairing: the *even* A register of each pair carries rows 0-7 and the *odd* one rows
// 8-15, while the K half is selected by the register index (0/1 -> first half, 2/3 -> second).
// Getting that mapping wrong silently transposes the accumulation, so it is pinned by
// tests/mma_emulation_test.cu, which runs every lowering against the shape it replaces on
// hardware that has both.
// =============================================================================================

// Turing-native fp16 tensor-core shape: 16x8x8.
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native fp16 shape with an fp16 accumulator, same lane geometry as m16n8k8 above.
__device__ __forceinline__ void mma_f16_m16n8k8_f16acc(unsigned& c0, unsigned& c1, unsigned a0,
                                                       unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native int8 tensor-core shape: 8x8x16.
__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}

// bf16 has no tensor-core path before sm_80. The pair below reads the bf16 payload straight out
// of a packed register (the high half of a float and the high 16 bits respectively), which is
// what the shuffle-FMA lowerings of mma_bf16 and mma_tf32_bits are built on.
__device__ __forceinline__ float bf16_low_to_float(unsigned packed) {
    return __uint_as_float(packed << 16);
}
__device__ __forceinline__ float bf16_high_to_float(unsigned packed) {
    return __uint_as_float(packed & 0xffff0000U);
}
#endif // NINFER_SM75

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM75)
    // No bf16 HMMA on Turing: evaluate the 16x8x16 tile with warp shuffles and fp32 FMAs. Each
    // lane reconstructs a full row of A and the two B columns it needs from its peers, so the
    // arithmetic is a straight fp32 dot product over K done in the fragment's own register
    // order. bf16 -> fp32 is exact (the payload is the fp32 high half), so the only difference
    // from the Ampere path is the summation order.
    const int lane    = threadIdx.x & 31;
    const int r       = lane >> 2;
    const int t       = lane & 3;
    const int base_a  = 4 * r;
    const int base_b0 = 8 * t;
    const int base_b1 = 8 * t + 4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
        const unsigned reg_a0 = __shfl_sync(0xffffffff, a0, base_a + k_idx);
        const unsigned reg_a1 = __shfl_sync(0xffffffff, a1, base_a + k_idx);
        const unsigned reg_a2 = __shfl_sync(0xffffffff, a2, base_a + k_idx);
        const unsigned reg_a3 = __shfl_sync(0xffffffff, a3, base_a + k_idx);

        const unsigned reg_b0_n0 = __shfl_sync(0xffffffff, b0, base_b0 + k_idx);
        const unsigned reg_b1_n0 = __shfl_sync(0xffffffff, b1, base_b0 + k_idx);
        const unsigned reg_b0_n1 = __shfl_sync(0xffffffff, b0, base_b1 + k_idx);
        const unsigned reg_b1_n1 = __shfl_sync(0xffffffff, b1, base_b1 + k_idx);

        const float a_r_k0  = bf16_low_to_float(reg_a0);
        const float a_r_k1  = bf16_high_to_float(reg_a0);
        const float a_r8_k0 = bf16_low_to_float(reg_a1);
        const float a_r8_k1 = bf16_high_to_float(reg_a1);

        const float a_r_k8  = bf16_low_to_float(reg_a2);
        const float a_r_k9  = bf16_high_to_float(reg_a2);
        const float a_r8_k8 = bf16_low_to_float(reg_a3);
        const float a_r8_k9 = bf16_high_to_float(reg_a3);

        const float b_n0_k0 = bf16_low_to_float(reg_b0_n0);
        const float b_n0_k1 = bf16_high_to_float(reg_b0_n0);
        const float b_n0_k8 = bf16_low_to_float(reg_b1_n0);
        const float b_n0_k9 = bf16_high_to_float(reg_b1_n0);

        const float b_n1_k0 = bf16_low_to_float(reg_b0_n1);
        const float b_n1_k1 = bf16_high_to_float(reg_b0_n1);
        const float b_n1_k8 = bf16_low_to_float(reg_b1_n1);
        const float b_n1_k9 = bf16_high_to_float(reg_b1_n1);

        sum0 += a_r_k0 * b_n0_k0 + a_r_k1 * b_n0_k1 + a_r_k8 * b_n0_k8 + a_r_k9 * b_n0_k9;
        sum1 += a_r_k0 * b_n1_k0 + a_r_k1 * b_n1_k1 + a_r_k8 * b_n1_k8 + a_r_k9 * b_n1_k9;
        sum2 += a_r8_k0 * b_n0_k0 + a_r8_k1 * b_n0_k1 + a_r8_k8 * b_n0_k8 + a_r8_k9 * b_n0_k9;
        sum3 += a_r8_k0 * b_n1_k0 + a_r8_k1 * b_n1_k1 + a_r8_k8 * b_n1_k8 + a_r8_k9 * b_n1_k9;
    }

    c0 += sum0;
    c1 += sum1;
    c2 += sum2;
    c3 += sum3;
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(NINFER_SM75)
    // Split the K=16 tile into its two K=8 halves: (a0, a1) x b0 is K 0-7 and (a2, a3) x b1 is
    // K 8-15. Both halves accumulate into the same 16x8 fp32 tile.
    mma_f16_m16n8k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16-accumulate variant: full-rate on consumer parts where f32-acc HMMA runs at
// half rate. D/C are two packed half2 registers: c0 = rows 0-7 column pair,
// c1 = rows 8-15 column pair of the same fragment the f32 variant returns in c0..c3.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& c0, unsigned& c1, unsigned a0, unsigned a1,
                                               unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // Same K split as mma_f16 above; the two partial sums now accumulate in fp16, so the
    // result differs from the single m16n8k16 f16-acc instruction by one rounding step.
    mma_f16_m16n8k8_f16acc(c0, c1, a0, a1, b0);
    mma_f16_m16n8k8_f16acc(c0, c1, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // 16x8x32 is four 8x8x16 tiles: the M halves are selected by the even/odd A register inside
    // each K half, so each (A, B) pair lands in the C half that names the same rows.
    mma_s8_m8n8k16(c0, c1, a0, b0); // rows 0-7,  K 0-15
    mma_s8_m8n8k16(c2, c3, a1, b0); // rows 8-15, K 0-15
    mma_s8_m8n8k16(c0, c1, a2, b1); // rows 0-7,  K 16-31
    mma_s8_m8n8k16(c2, c3, a3, b1); // rows 8-15, K 16-31
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // Turing's tensor cores stop at fp16/int8: there is no FP8 shape to lower onto. The FP8
    // kernels are excluded from a 75 build (see src/CMakeLists.txt), so this is a tripwire for
    // a future caller rather than a silent zero.
    (void)c0;
    (void)c1;
    (void)c2;
    (void)c3;
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    __trap();
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM120 (Blackwell): the unified f8f6f4 kind covers e4m3 x e4m3.
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // sm_89 (Ada) and earlier: FP8 mma.sync carries no .kind modifier (PTX ISA 7.8+);
    // identical m16n8k32 fragment layout (A = 4 x .b32, B = 2 x .b32, C = 4 x .f32).
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // TF32 tensor cores arrive with sm_80. Evaluate the 16x8x8 tile with warp shuffles and fp32
    // FMAs instead. Callers hand in operands already rounded to tf32, and the product of two
    // tf32 values is exact in fp32 (11-bit significands), so this reproduces the tensor-core
    // result up to the summation order.
    const int lane    = threadIdx.x & 31;
    const int r       = lane >> 2;
    const int t       = lane & 3;
    const int base_a  = 4 * r;
    const int base_b0 = 8 * t;
    const int base_b1 = 8 * t + 4;

    const float fa0 = __uint_as_float(a0);
    const float fa1 = __uint_as_float(a1);
    const float fa2 = __uint_as_float(a2);
    const float fa3 = __uint_as_float(a3);
    const float fb0 = __uint_as_float(b0);
    const float fb1 = __uint_as_float(b1);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
        const float a_r_k   = __shfl_sync(0xffffffff, fa0, base_a + k_idx);
        const float a_r8_k  = __shfl_sync(0xffffffff, fa1, base_a + k_idx);
        const float a_r_k4  = __shfl_sync(0xffffffff, fa2, base_a + k_idx);
        const float a_r8_k4 = __shfl_sync(0xffffffff, fa3, base_a + k_idx);

        const float b_n0_k  = __shfl_sync(0xffffffff, fb0, base_b0 + k_idx);
        const float b_n0_k4 = __shfl_sync(0xffffffff, fb1, base_b0 + k_idx);
        const float b_n1_k  = __shfl_sync(0xffffffff, fb0, base_b1 + k_idx);
        const float b_n1_k4 = __shfl_sync(0xffffffff, fb1, base_b1 + k_idx);

        sum0 += a_r_k * b_n0_k + a_r_k4 * b_n0_k4;
        sum1 += a_r_k * b_n1_k + a_r_k4 * b_n1_k4;
        sum2 += a_r8_k * b_n0_k + a_r8_k4 * b_n0_k4;
        sum3 += a_r8_k * b_n1_k + a_r8_k4 * b_n1_k4;
    }

    c0 += sum0;
    c1 += sum1;
    c2 += sum2;
    c3 += sum3;
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops
