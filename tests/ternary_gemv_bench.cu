// Decode-shaped (T = 1) ternary GEMV benchmark: PTQ1_0 against PQ2_0 on the shapes the
// 27B model actually uses.
//
// Why this exists: the engine's whole-model PTQ1_0 decode ran at 0.44x the reference
// engine's while PQ2_0 ran at 0.69x, and the GEMV is the only thing that differs between
// the two formats (same dispatch, same launch geometry, same non-ternary kernels).
// Measuring the kernels side by side at production shapes separates "the GEMV is slow"
// from "the engine around it is slow" without a 9-minute engine rebuild per experiment.
//
// Run: builder.exe -C . -test tests/ternary_gemv_bench.cu
//
// The packing is what the kernels care about, not semantic validity, so the payloads are
// filled with a deterministic byte stream: both formats decode *something* and the timing
// and the byte traffic are exact.

#include "ops/linear/ternary/ternary_rowsplit_gemv_ptq1.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ninfer::ops::detail;

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::printf("FAIL %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

struct Shape {
    const char* name;
    std::int32_t n;
    std::int32_t k;
};

// K is the contiguous axis and is a multiple of the 128-weight group; the widths are the
// ones the model's ternary linears use.
constexpr Shape kShapes[] = {
    {"mlp-up/gate 17408x5120", 17408, 5120},
    {"mlp-down    5120x17408", 5120, 17408},
    {"attn-qkv    6144x5120", 6144, 5120},
    {"mlp-mid     17408x10240", 17408, 10240},
    {"lm-head     248320x5120", 248320, 5120},
};

constexpr int kWarmup = 3;
constexpr int kIters  = 30;

struct Buffers {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* highs  = nullptr;
    std::uint8_t* scales = nullptr;
    __nv_bfloat16* x     = nullptr;
    __nv_bfloat16* out   = nullptr;
    std::size_t code_bytes  = 0;
    std::size_t high_bytes  = 0;
    std::size_t scale_bytes = 0;
    std::int64_t rows       = 0;
};

std::vector<std::uint8_t> payload(std::size_t bytes, std::uint32_t seed) {
    std::vector<std::uint8_t> out(bytes);
    std::uint32_t state = seed | 1u;
    for (std::size_t i = 0; i < bytes; ++i) {
        state = state * 1664525u + 1013904223u;
        out[i] = static_cast<std::uint8_t>(state >> 24);
    }
    return out;
}

// One warp per output row, as the production launchers do.
constexpr unsigned grid_for(std::int64_t rows) {
    return static_cast<unsigned>((rows + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock);
}

// Registers decide the resident-warp count for this kernel family, so report them next to
// the timing: a change that buys instructions but costs warps is not a win.
void report_regs(const void* fn, const char* label) {
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, fn) != cudaSuccess) { return; }
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, kGemvWarpsPerBlock * 32, 0);
    std::printf("    %-6s regs=%d  blocks/SM=%d  warps/SM=%d\n", label, attr.numRegs,
                blocks_per_sm, blocks_per_sm * kGemvWarpsPerBlock);
}

void launch_ptq1(const Buffers& b, std::int32_t groups) {
    ternary_ptq1_gemv_kernel<<<grid_for(b.rows), kGemvWarpsPerBlock * 32>>>(
        b.x, b.codes, b.highs, b.scales, b.out, static_cast<std::int32_t>(b.rows), groups);
}

void launch_pq2(const Buffers& b, std::int32_t groups) {
    ternary_pq2_gemv_kernel<<<grid_for(b.rows), kGemvWarpsPerBlock * 32>>>(
        b.x, b.codes, b.scales, b.out, static_cast<std::int32_t>(b.rows), groups);
}

double time_kernel(void (*launch)(const Buffers&, std::int32_t), const Buffers& b,
                   std::int32_t groups) {
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    for (int i = 0; i < kWarmup; ++i) { launch(b, groups); }
    check(cudaDeviceSynchronize(), "warmup sync");
    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < kIters; ++i) { launch(b, groups); }
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / kIters;
}

} // namespace

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::printf("no CUDA device\n");
        return 77;
    }
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "device props");
    std::printf("device: %s (sm_%d%d, %.1f GiB, %.1f GB/s peak)\n", prop.name, prop.major,
                prop.minor, prop.totalGlobalMem / 1073741824.0,
                2.0 * prop.memoryClockRate * 1e-6 * (prop.memoryBusWidth / 8.0));
    std::printf("%-24s %6s | %9s %9s | %9s %9s | %s\n", "shape", "gpr", "ptq1 ms", "ptq1 GB/s",
                "pq2 ms", "pq2 GB/s", "ptq1/pq2");

    for (const Shape& s : kShapes) {
        const std::int32_t groups = s.k / 128;
        Buffers ptq1;
        ptq1.rows       = s.n;
        ptq1.code_bytes = static_cast<std::size_t>(s.n) * groups * kPTQ1CodeBytes;
        ptq1.high_bytes = static_cast<std::size_t>(s.n) * groups * kPTQ1HighBytes;
        ptq1.scale_bytes = static_cast<std::size_t>(s.n) * groups * kGemvScaleBytesPerGroup;
        check(cudaMalloc(&ptq1.x, static_cast<std::size_t>(s.k) * 2), "malloc x");
        check(cudaMalloc(&ptq1.codes, ptq1.code_bytes), "malloc ptq1 codes");
        check(cudaMalloc(&ptq1.highs, ptq1.high_bytes), "malloc ptq1 highs");
        check(cudaMalloc(&ptq1.scales, ptq1.scale_bytes), "malloc ptq1 scales");
        check(cudaMalloc(&ptq1.out, static_cast<std::size_t>(s.n) * 2), "malloc ptq1 out");
        {
            const auto codes = payload(ptq1.code_bytes, 7u);
            const auto highs = payload(ptq1.high_bytes, 11u);
            const auto scale = payload(ptq1.scale_bytes, 13u);
            const auto act   = payload(static_cast<std::size_t>(s.k) * 2, 17u);
            check(cudaMemcpy(ptq1.codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "h2d codes");
            check(cudaMemcpy(ptq1.highs, highs.data(), highs.size(), cudaMemcpyHostToDevice), "h2d highs");
            check(cudaMemcpy(ptq1.scales, scale.data(), scale.size(), cudaMemcpyHostToDevice), "h2d scales");
            check(cudaMemcpy(ptq1.x, act.data(), act.size(), cudaMemcpyHostToDevice), "h2d x");
        }

        // PQ2_0 is the same geometry with a 32-byte code plane and no high plane.
        Buffers pq2;
        pq2.rows       = s.n;
        pq2.code_bytes = static_cast<std::size_t>(s.n) * groups * kGemvCodeBytesPerGroup;
        pq2.scale_bytes = ptq1.scale_bytes;
        check(cudaMalloc(&pq2.codes, pq2.code_bytes), "malloc pq2 codes");
        check(cudaMalloc(&pq2.scales, pq2.scale_bytes), "malloc pq2 scales");
        check(cudaMalloc(&pq2.out, static_cast<std::size_t>(s.n) * 2), "malloc pq2 out");
        {
            const auto codes = payload(pq2.code_bytes, 19u);
            const auto scale = payload(pq2.scale_bytes, 23u);
            check(cudaMemcpy(pq2.codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "h2d pq2 codes");
            check(cudaMemcpy(pq2.scales, scale.data(), scale.size(), cudaMemcpyHostToDevice), "h2d pq2 scales");
        }
        pq2.x = ptq1.x;

        const double ptq1_ms = time_kernel(launch_ptq1, ptq1, groups);
        const double pq2_ms  = time_kernel(launch_pq2, pq2, groups);
        const double ptq1_gb =
            (ptq1.code_bytes + ptq1.high_bytes + ptq1.scale_bytes) / (ptq1_ms * 1e6);
        const double pq2_gb = (pq2.code_bytes + pq2.scale_bytes) / (pq2_ms * 1e6);
        std::printf("%-24s %6d | %9.3f %9.1f | %9.3f %9.1f | %.2f\n", s.name, groups, ptq1_ms,
                    ptq1_gb, pq2_ms, pq2_gb, pq2_ms / ptq1_ms);
        report_regs(reinterpret_cast<const void*>(ternary_ptq1_gemv_kernel), "ptq1");
        report_regs(reinterpret_cast<const void*>(ternary_pq2_gemv_kernel), "pq2");

        cudaFree(ptq1.codes);
        cudaFree(ptq1.highs);
        cudaFree(ptq1.scales);
        cudaFree(ptq1.x);
        cudaFree(ptq1.out);
        cudaFree(pq2.codes);
        cudaFree(pq2.scales);
        cudaFree(pq2.out);
    }
    return 0;
}
