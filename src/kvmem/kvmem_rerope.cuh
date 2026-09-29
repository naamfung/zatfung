// MODIFIED for zatfung (zat6 疾 fung1 风 引擎).
// KVMem K2: in-place K phase re-rotation for int8-group64 cached pages.
//
// When the KVMem window compacts selected blocks into new slots, every moved
// block's cached K carries RoPE phases of its OLD window slot while the
// attention kernel will read it at the NEW slot (the kernel derives positions
// from block-table indices). This kernel re-bakes the phases:
//
//   stored   = Quant_g(H256(RoPE_src(K_raw)))
//   re-baked = Quant_g(H256(RoPE_dst(H256(Dequant_g(stored)))))
//
// The stored domain is the FULL-ROW normalized H256 rotation -- the exact
// convention of the production append kernels (kv_cache_append_full_i8_kernel /
// kv_cache_append_full_i8_page_kernel), of the attention Q path and of the KVMem
// scorer. H256 is an involution, so de-rotation and re-rotation share
// normalized_hadamard_d256_inplace. Because that transform mixes all four 64-dim
// groups, re-baking rewrites every group's codes and scales; treating the page as
// a per-group H64 and rewriting group 0 alone -- as an earlier revision of this
// kernel did -- corrupts the moved page. The rotary pairs only exist in the RAW
// domain, which is why the whole row is de-rotated first.
//
// RoPE here is Qwen3.6 Text MRoPE: rotary_dim 64, split-half pairs (p, p+32)
// for p in [0,32), angle phi = positions[axis(p)] * kTextRopeInvFrequency[p],
// axis = p % 3 (src/ops/kernel/rope.cuh). After the H256 undo the raw rotary pair
// (p, p+32) sits at (row[0], row[1]) of lane p -- the same lane layout the KVMem
// scoring kernel uses for its un-RoPE.
//
// One warp owns one (page, token, head): lane l holds dims l + 32*r in row[r].
// Everything uses the exact production helpers (normalized_hadamard_d256_inplace,
// kv_cache_int8_quant_*) so a re-baked page is bit-consistent with what the append
// path would have written for the same raw K at the new position (up to fp
// reassociation).

#pragma once

#include <cuda_runtime.h>

#include <cstdio>

#include "ops/kernel/rope.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::kvmem {

// Local check: kvmem stays self-contained (core/device.h carries the SM gate).
inline void kvmem_cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", expr, file, line,
                     cudaGetErrorString(err));
    }
}

#define KVMEM_CUDA_CHECK(expr) ::ninfer::kvmem::kvmem_cuda_check((expr), #expr, __FILE__, __LINE__)

// The re-phase pipeline is intentionally built from the EXACT production helpers
// (codec, Hadamard, RoPE tables) so a re-baked page is bit-consistent with what
// the append path writes for the same raw K at the new position.
using namespace ninfer::ops;

// One warp re-bakes one (page, token, head).
//   codes/scales : the device page (page-major layout of paged_kv_element_offset)
//   src_pos/dst_pos : [axis * 64 + token] absolute MRoPE positions, page-local
//                     token index (layout matches the engine's positions[T,3]
//                     with T narrowed to one page)
//   valid_tokens : tokens < valid_tokens are re-baked (tail pages).
template <int KVHeads>
__global__ void kvmem_rerope_int8_g64_kernel(std::int8_t* codes, __half* scales,
                                             const std::int32_t* src_pos,
                                             const std::int32_t* dst_pos, int pages,
                                             int valid_tokens) {
    const int token = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    const int page  = static_cast<int>(blockIdx.z);
    const int lane  = static_cast<int>(threadIdx.x);  // 32 lanes == 32 RoPE pairs
    if (token >= valid_tokens) { return; }

    const std::int64_t code_base =
        paged_kv_element_offset<kKVCacheInt8HeadDim, KVHeads>(page, head, token, 0);
    const std::int64_t scale_base =
        paged_kv_element_offset<kKVCacheInt8Groups, KVHeads>(page, head, token, 0);

    // -- dequant the full row (dims lane + 32*r), then undo the H256 rotation --
    float row[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d   = lane + 32 * r;
        const float s = __half2float(scales[scale_base + r / 2]);
        row[r]        = static_cast<float>(codes[code_base + d]) * s;
    }
    normalized_hadamard_d256_inplace(row, lane);

    // -- un-RoPE at the baked slot, re-RoPE at the new slot --
    // The raw rotary pair (p, p+32) sits at (row[0], row[1]) of lane p.
    const int axis = lane % 3;
    const float phi_src =
        static_cast<float>(src_pos[axis * valid_tokens + token]) * kTextRopeInvFrequency[lane];
    const float phi_dst =
        static_cast<float>(dst_pos[axis * valid_tokens + token]) * kTextRopeInvFrequency[lane];
    float ss = 0.0f, cs = 0.0f, sd = 0.0f, cd = 0.0f;
    sincosf(phi_src, &ss, &cs);
    sincosf(phi_dst, &sd, &cd);
    // stored pair convention (apply_rope_head): x0' = x0*cos - x1*sin; x1' = x1*cos + x0*sin
    const float u0 = row[0] * cs + row[1] * ss;  // inverse: sin flips sign
    const float u1 = row[1] * cs - row[0] * ss;
    row[0]         = u0 * cd - u1 * sd;
    row[1]         = u1 * cd + u0 * sd;

    // -- re-apply the H256 rotation and re-quantize every group --
    normalized_hadamard_d256_inplace(row, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        float absmax = fmaxf(fabsf(row[2 * group]), fabsf(row[2 * group + 1]));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, offset));
        }
        const KVCacheInt8QuantParams qp = kv_cache_int8_quant_params(absmax);
        const int d0                   = group * kKVCacheInt8Group + lane;
        const int d1                   = d0 + 32;
        codes[code_base + d0] = kv_cache_int8_quant_code(row[2 * group], qp.inverse_scale);
        codes[code_base + d1] = kv_cache_int8_quant_code(row[2 * group + 1], qp.inverse_scale);
        if (lane == 0) { scales[scale_base + group] = qp.scale; }
    }
}

// Launcher: re-bakes `pages` whole pages (all 64 tokens). A partially filled
// tail page passes valid_tokens < 64 for that page only by launching it
// separately; the production executor (K1b) will own that split.
template <int KVHeads>
void kvmem_rerope_int8_g64_pages(std::int8_t* codes, __half* scales, int pages,
                                 const std::int32_t* src_pos, const std::int32_t* dst_pos,
                                 cudaStream_t stream) {
    static_assert(kKVCacheInt8HeadDim == 256, "re-phase assumes the 256-dim MRoPE head");
    const dim3 grid(kPagedKVPageSize, KVHeads, pages);
    const dim3 block(32, 1u, 1u);
    kvmem_rerope_int8_g64_kernel<KVHeads>
        <<<grid, block, 0, stream>>>(codes, scales, src_pos, dst_pos, pages, kPagedKVPageSize);
    KVMEM_CUDA_CHECK(cudaGetLastError());
}

// Per-plane variant: the executor addresses one layer's K-codes / K-scales plane
// tensors directly. `page_index` is the physical page within the plane (planes
// are page-major: page stride = 256 * 64 * KVHeads for codes, 4 * 64 * KVHeads
// for scales). src/dst positions are page-local [axis * 64 + token].
template <int KVHeads>
void kvmem_rerope_int8_g64_plane(std::int8_t* codes_plane, __half* scales_plane,
                                 std::int32_t page_index, const std::int32_t* src_pos,
                                 const std::int32_t* dst_pos, cudaStream_t stream) {
    constexpr std::int64_t kCodeStride =
        static_cast<std::int64_t>(kKVCacheInt8HeadDim) * kPagedKVPageSize * KVHeads;
    constexpr std::int64_t kScaleStride =
        static_cast<std::int64_t>(kKVCacheInt8Groups) * kPagedKVPageSize * KVHeads;
    const dim3 grid(kPagedKVPageSize, KVHeads, 1);
    const dim3 block(32, 1u, 1u);
    kvmem_rerope_int8_g64_kernel<KVHeads><<<grid, block, 0, stream>>>(
        codes_plane + static_cast<std::int64_t>(page_index) * kCodeStride,
        scales_plane + static_cast<std::int64_t>(page_index) * kScaleStride, src_pos, dst_pos, 1,
        kPagedKVPageSize);
    KVMEM_CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kvmem
