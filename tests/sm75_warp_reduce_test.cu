// Standalone verification for the sm_75 fallback of `__reduce_max_sync` in
// ops/kernel/sampling_device.cuh (build and run with
// `builder.exe -test tests/sm75_warp_reduce_test.cu`).
//
// `__reduce_max_sync` is sm_80+, so the sampler's tile merge lowers it to a shuffle butterfly.
// The subtle part is the mask: the call sites reduce over only the low `kSamplingTileWarps`
// lanes, and the lanes outside the mask hold arbitrary values that must not be folded in.
//
// NINFER_SM75 is defined before the include, so the helper under test is the production one.
// The check runs on an sm_80+ device, where the instruction being replaced also exists; both
// are fed the same per-lane values and compared lane by lane.
//
// One extra case is asserted: a non-contiguous mask, which the production code does not use but
// which shows the lowering honours the mask generally rather than only for a low-bit run.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Selects the sm_75 lowering of the helper.
#define NINFER_SM75 1
#undef NINFER_SM86
#undef NINFER_SM89
#include "ops/kernel/sampling_device.cuh"

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

__global__ void k_reduce(const unsigned* values, unsigned mask, unsigned* hardware,
                         unsigned* lowering) {
    const int lane    = threadIdx.x & 31;
    const unsigned in = values[lane];
    // Both the intrinsic and the shuffle lowering require exactly the mask lanes to take part,
    // which is how the production call sites use them (`if (lane < kSamplingTileWarps)`).
    if (((mask >> lane) & 1u) == 0u) { return; }
    lowering[lane] = ninfer::ops::sampling_warp_max(mask, in);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    hardware[lane] = __reduce_max_sync(mask, in);
#else
    hardware[lane] = 0u;
#endif
}

int device_cc_major() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { return 0; }
    return prop.major;
}

} // namespace

int main() {
    if (device_cc_major() < 8) {
        std::printf("device is not sm_80+; nothing to compare against\n");
        std::printf("failures=0  VERDICT: SKIP\n");
        return 0;
    }

    // Contiguous low-bit masks mirror the production `(1u << kSamplingTileWarps) - 1u`; the
    // interleaved one shows the mask is honoured in general.
    const unsigned masks[] = {0xffffffffu, 0x0000ffffu, 0x000000ffu, 0x0000000fu,
                              0x00000003u, 0x00000001u,   0x55555555u};

    std::mt19937 rng(0x5eedu);
    std::uniform_int_distribution<unsigned> dist(0u, 0xffffffffu);

    std::vector<unsigned> host(32);
    unsigned* dev_values = nullptr;
    unsigned* dev_hw     = nullptr;
    unsigned* dev_em     = nullptr;
    if (cudaMalloc(&dev_values, 32 * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc(&dev_hw, 32 * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc(&dev_em, 32 * sizeof(unsigned)) != cudaSuccess) {
        std::printf("cudaMalloc failed\n");
        return 2;
    }

    std::vector<unsigned> hw(32), em(32);
    std::vector<unsigned> expected(32);

    for (int round = 0; round < 16; ++round) {
        for (unsigned& v : host) { v = dist(rng); }
        if (cudaMemcpy(dev_values, host.data(), 32 * sizeof(unsigned), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            std::printf("cudaMemcpy failed\n");
            return 2;
        }

        for (const unsigned mask : masks) {
            k_reduce<<<1, 32>>>(dev_values, mask, dev_hw, dev_em);
            const cudaError_t sync = cudaDeviceSynchronize();
            if (sync != cudaSuccess) {
                std::printf("kernel launch failed for mask 0x%08x: %s (%s)\n", mask,
                            cudaGetErrorName(sync), cudaGetErrorString(sync));
                return 2;
            }
            if (cudaMemcpy(hw.data(), dev_hw, 32 * sizeof(unsigned), cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
                cudaMemcpy(em.data(), dev_em, 32 * sizeof(unsigned), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                std::printf("cudaMemcpy D2H failed\n");
                return 2;
            }

            // Only the mask lanes have a defined result, in either implementation.
            unsigned want = 0u;
            for (int lane = 0; lane < 32; ++lane) {
                if (((mask >> lane) & 1u) != 0u) { want = want > host[lane] ? want : host[lane]; }
            }

            int bad = 0;
            for (int lane = 0; lane < 32; ++lane) {
                if (((mask >> lane) & 1u) == 0u) { continue; }
                if (em[lane] != want || hw[lane] != want) { ++bad; }
            }
            int mask_lanes = 0;
            for (int lane = 0; lane < 32; ++lane) {
                if (((mask >> lane) & 1u) != 0u) { ++mask_lanes; }
            }
            CHECK(bad == 0, "mask 0x%08x: %d of %d mask lanes wrong", mask, bad, mask_lanes);
        }
    }

    cudaFree(dev_values);
    cudaFree(dev_hw);
    cudaFree(dev_em);

    std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
