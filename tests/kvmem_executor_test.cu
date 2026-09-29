// Executor integration test for KVMem K1b (standalone; build with
// `builder.exe -test tests/kvmem_executor_test.cu`).
//
// Exercises the real machinery the window executor will drive:
//   DeviceKVPagePool (materialize / release / per-plane addressing)
//   HostKVArena      (evict target, geometry-matched)
//   copy_to_host / copy_from_host (whole page groups, all layers)
//   kvmem_rerope     (in-place K phase re-bake on pool-resident planes)
//
// Flow: build the full expected page-group content on the HOST (K group 0 is a
// real quantized-at-src block per (page, layer, head, token); everything else is
// deterministic filler), upload, evict pages 0..3 to the host arena, release
// their device pages, stage the blocks back into 4 fresh pages in REVERSE order
// (real compaction moves slots), re-phase each to its new window slot, and
// verify: position-free data bitwise, re-baked group 0 within one code step.
// All device access goes through cudaMemcpy / pool copies / kernels.
#include "core/host_kv_arena.h"
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

constexpr int kLayers = 2;
constexpr int kHeads  = 2;
constexpr int kPages  = 6;
constexpr int kEvict  = 4;
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

float host_freq(int pair) { return std::pow(1e7f, -2.0f * static_cast<float>(pair) / 64.0f); }

// Host mirror of normalized_hadamard_d256_inplace (dims indexed d = l + 32*r): the
// production H256 quantization rotation the append path applies to K.
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

// Plane addressing (page-major): element (head, token, d) of page P sits at
// plane_base + P * elems_per_page + leading*(head*64 + token) + d, where the
// offset is in ELEMENTS of the plane dtype (I8 = 1 byte, FP16 = 2 bytes).
std::int64_t elems_per_page(int leading) {
    return static_cast<std::int64_t>(leading) * kTokens * kHeads;
}
std::int64_t in_page_elems(int leading, int head, int token, int d) {
    return static_cast<std::int64_t>(leading) * (static_cast<std::int64_t>(head) * kTokens + token) +
           d;
}

struct KRow {
    std::int8_t codes[256];
    unsigned short scale_bits[4];
};

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::srand(777);
    try {
        std::printf("enter main\n");

        // ---- geometry: 2 layers x [K-codes, V-codes, K-scale, V-scale] ----
        KVPageGeometry geometry;
        geometry.page_tokens        = kTokens;
        geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;  // matches the text cache
        for (int layer = 0; layer < kLayers; ++layer) {
            geometry.planes.push_back({DType::I8, 256, kHeads, 256});
            geometry.planes.push_back({DType::I8, 256, kHeads, 256});
            geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
            geometry.planes.push_back({DType::FP16, 4, kHeads, 256});
        }
        constexpr int kTotalPlanes = kLayers * 4;

        LayoutBuilder builder;
        const DeviceKVPagePoolLayout pool_layout =
            plan_device_kv_page_pool(builder, DeviceKVPagePoolSpec{.page_group_count = kPages,
                                                                   .geometry = geometry});
        const KVExecutionTableLayout tables_layout =
            plan_kv_execution_tables(builder, KVExecutionTableSpec{.logical_page_capacity = kPages,
                                                                   .table_rows = 1});
        const std::size_t backing_bytes =
            pool_layout.payload_bytes() + tables_layout.metadata_bytes() + (1u << 20);
        void* backing = nullptr;
        CHECK(cudaMalloc(&backing, backing_bytes) == cudaSuccess, "cudaMalloc backing");
        std::printf("planned; backing %.1f MiB\n", backing_bytes / 1048576.0);

        DeviceKVPagePool pool(DeviceSpan{backing, backing_bytes}, pool_layout);
        KVExecutionTablePool tables(DeviceSpan{backing, backing_bytes}, tables_layout, pool);
        std::printf("pool + tables ok\n");

        std::vector<Tensor> planes;
        // Per-page stride is counted in ELEMENTS of the plane dtype:
        std::vector<std::int64_t> plane_strides(static_cast<std::size_t>(kTotalPlanes));
        for (int p = 0; p < kTotalPlanes; ++p) {
            planes.push_back(pool.plane(static_cast<std::size_t>(p)));
            plane_strides[static_cast<std::size_t>(p)] = elems_per_page(
                geometry.planes[static_cast<std::size_t>(p)].leading_extent);
        }

        // ---- host arena (evict target), geometry-matched to the pool ----
        const HostKVPageLayout host_layout = plan_host_kv_page_layout(geometry);
        const std::vector<HostKVPageLayout> supported{host_layout};
        HostKVArena arena(host_layout.page_stride * static_cast<std::size_t>(kEvict) + (1u << 20),
                          supported);
        auto host_alloc = arena.allocate(host_layout, static_cast<std::uint32_t>(kEvict));
        CHECK(host_alloc.has_value(), "host allocation failed");
        HostKVAllocationView host_view = arena.writable_view(*host_alloc);

        // ---- build the expected content on the HOST, plane by plane ----
        std::vector<std::vector<std::uint8_t>> host_plane(static_cast<std::size_t>(kTotalPlanes));
        std::vector<std::int64_t> plane_bytes_total(static_cast<std::size_t>(kTotalPlanes));
        for (int p = 0; p < kTotalPlanes; ++p) {
            const int leading = geometry.planes[static_cast<std::size_t>(p)].leading_extent;
            const int elem = geometry.planes[static_cast<std::size_t>(p)].dtype ==
                                     DType::FP16
                                 ? 2
                                 : 1;
            const std::int64_t bytes = elems_per_page(leading) * elem * kPages;
            plane_bytes_total[static_cast<std::size_t>(p)] = bytes;
            host_plane[static_cast<std::size_t>(p)].resize(static_cast<std::size_t>(bytes));
            auto* base = host_plane[static_cast<std::size_t>(p)].data();
            const std::int64_t page = bytes / kPages;
            for (std::int64_t b = 0; b < bytes; ++b) {
                base[b] = static_cast<std::uint8_t>(static_cast<int>(b / page) * 131 +
                                                    p * 17 + static_cast<int>(b % page)) *
                              31 +
                          7;
            }
        }
        // K full row per (page, layer, head, token): raw -> rope -> H256 -> per-group
        // quantize. This mirrors the PRODUCTION append pipeline (full-row H256 rotation,
        // per-group-of-64 FP16 scales) -- the convention the re-phase kernel must assume.
        std::vector<KRow> model(static_cast<std::size_t>(kPages) * kLayers * kHeads * kTokens);
        std::vector<float> row(256);
        for (int i = 0; i < kPages; ++i) {
            for (int l = 0; l < kLayers; ++l) {
                const std::size_t kc_plane = static_cast<std::size_t>(l * 4);
                const std::size_t ks_plane = static_cast<std::size_t>(l * 4 + 2);
                const std::size_t kc_page  = static_cast<std::size_t>(elems_per_page(256) * i);
                // FP16 plane: 2 bytes per element -- the page offset is in BYTES.
                const std::size_t ks_page  = static_cast<std::size_t>(elems_per_page(4) * i) * 2;
                for (int h = 0; h < kHeads; ++h) {
                    for (int t = 0; t < kTokens; ++t) {
                        const std::int32_t pos3[3] = {100000 + i * kTokens + t,
                                                      100000 + i * kTokens + t,
                                                      100000 + i * kTokens + t};
                        for (int d = 0; d < 256; ++d) {
                            row[d] = static_cast<float>(std::rand() % 2001 - 1000) / 1000.0f;
                        }
                        rope_group0(row.data(), pos3, false);
                        hadamard256_host(row.data());
                        KRow out;
                        for (int g = 0; g < kGroups; ++g) {
                            float absmax = 0.0f;
                            for (int d = 0; d < 64; ++d) {
                                absmax = std::max(absmax, std::fabs(row[g * 64 + d]));
                            }
                            const unsigned short bits = __half_as_ushort(
                                __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
                            const float s = __half2float(__ushort_as_half(bits));
                            const float inv = s > 0.0f ? 1.0f / s : 0.0f;
                            for (int d = 0; d < 64; ++d) {
                                out.codes[g * 64 + d] = static_cast<std::int8_t>(std::max(
                                    -127,
                                    std::min(127,
                                             static_cast<int>(std::lrintf(row[g * 64 + d] * inv)))));
                            }
                            out.scale_bits[g] = bits;
                        }
                        model[(static_cast<std::size_t>(i) * kLayers + l) * kHeads * kTokens +
                              static_cast<std::size_t>(h) * kTokens + t] = out;
                        auto* kc = reinterpret_cast<std::int8_t*>(
                                       host_plane[kc_plane].data() + kc_page) +
                                   in_page_elems(256, h, t, 0);
                        for (int d = 0; d < 256; ++d) { kc[d] = out.codes[d]; }
                        auto* ks = reinterpret_cast<unsigned short*>(
                            host_plane[ks_plane].data() + ks_page) +
                                   in_page_elems(4, h, t, 0);
                        for (int g = 0; g < kGroups; ++g) { ks[g] = out.scale_bits[g]; }
                    }
                }
            }
        }
        std::printf("host content built\n");

        // ---- upload, materialize 6 device pages ----
        for (int p = 0; p < kTotalPlanes; ++p) {
            CHECK(cudaMemcpy(planes[static_cast<std::size_t>(p)].data,
                             host_plane[static_cast<std::size_t>(p)].data(),
                             host_plane[static_cast<std::size_t>(p)].size(),
                             cudaMemcpyHostToDevice) == cudaSuccess,
                  "H2D plane %d", p);
        }
        auto reservation = pool.reserve(kPages);
        CHECK(reservation.has_value(), "reserve 6 pages failed");
        std::vector<DeviceKVPageLease> leases;
        leases.reserve(kPages);
        pool.materialize(*reservation, kPages, leases);
        std::vector<std::int32_t> page_index(kPages);
        for (int i = 0; i < kPages; ++i) {
            page_index[i] = pool.physical_index_of(leases[i].handle());
        }
        std::printf("materialized 6 pages\n");

        // ---- evict pages 0..3 to the host arena ----
        std::vector<DeviceKVPageHandle> evict_handles;
        for (int i = 0; i < kEvict; ++i) { evict_handles.push_back(leases[i].handle()); }
        pool.copy_to_host(evict_handles, host_view);
        cudaStreamSynchronize(0);
        std::printf("post copy_to_host err=%s\n", cudaGetErrorString(cudaGetLastError()));
        std::printf("evicted 4 page groups to host\n");

        // ---- release their device pages, materialize fresh ones ----
        for (int i = 0; i < kEvict; ++i) { leases[i].release(); }
        auto fresh_reservation = pool.reserve(kEvict);
        CHECK(fresh_reservation.has_value(), "reserve 4 fresh pages failed");
        std::vector<DeviceKVPageLease> fresh;
        fresh.reserve(kEvict);
        pool.materialize(*fresh_reservation, kEvict, fresh);
        std::vector<std::int32_t> fresh_index(kEvict);
        for (int i = 0; i < kEvict; ++i) {
            fresh_index[i] = pool.physical_index_of(fresh[i].handle());
        }
        std::printf("fresh pages ready\n");

        // ---- stage back in REVERSE order: host slot i -> device slot (3-i) ----
        std::vector<DeviceKVPageHandle> fresh_handles;
        for (int i = 0; i < kEvict; ++i) { fresh_handles.push_back(fresh[i].handle()); }
        for (int i = 0; i < kEvict; ++i) {
            const int dst_slot = kEvict - 1 - i;
            pool.copy_from_host(host_view.subview(static_cast<std::uint32_t>(i), 1),
                                std::span<const DeviceKVPageHandle>(&fresh_handles[dst_slot], 1));
            cudaStreamSynchronize(0);
            std::printf("post copy_from_host %d err=%s\n", i,
                        cudaGetErrorString(cudaGetLastError()));
        }
        cudaStreamSynchronize(0);
        std::printf("staged 4 blocks back in reverse order\n");

        // DEBUG: snapshot group-0 K codes of slot 0 before re-phasing.
        {
            std::int8_t kc0[64];
            unsigned short sb;
            cudaMemcpy(kc0, static_cast<std::int8_t*>(planes[0].data) +
                                 fresh_index[0] * plane_strides[0],
                       64, cudaMemcpyDeviceToHost);
            cudaMemcpy(&sb, static_cast<unsigned short*>(planes[2].data) +
                                fresh_index[0] * plane_strides[2],
                       2, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            std::printf("pre-rerope slot0 l0 h0 t0 g0 codes:");
            for (int d = 0; d < 8; ++d) { std::printf(" %d", static_cast<int>(kc0[d])); }
            std::printf(" scale_bits=%04x scale=%f\n", sb,
                        __half2float(__ushort_as_half(sb)));
            // Input integrity: full 64-dim dequant vs the model for (3,l0,h0,t0).
            const KRow& mdl = model[(static_cast<std::size_t>(3) * kLayers + 0) * kHeads * kTokens + 0];
            const float ms = __half2float(__ushort_as_half(mdl.scale_bits[0]));
            double maxd = 0.0;
            for (int d = 0; d < 64; ++d) {
                const float dev_v = static_cast<float>(kc0[d]) * __half2float(__ushort_as_half(sb));
                const float mdl_v = static_cast<float>(mdl.codes[d]) * ms;
                maxd = std::max(maxd, static_cast<double>(std::fabs(dev_v - mdl_v)));
            }
            std::printf("input integrity max diff = %f (model scale %f)\n", maxd, ms);
            const KRow& want = model[(static_cast<std::size_t>(3) * kLayers + 0) * kHeads * kTokens + 0];
            std::printf("model(3,l0,h0,t0) g0 codes:");
            for (int d = 0; d < 8; ++d) { std::printf(" %d", static_cast<int>(want.codes[d])); }
            std::printf(" scale_bits=%04x\n", want.scale_bits[0]);
        }

        // ---- re-phase each staged page: src (absolute) -> new window slot ----
        // Axis-major layout [axis * 64 + token] -- matches the engine's positions
        // tensor and the kernel's read convention.
        std::vector<std::int32_t> src_pos(3 * kTokens), dst_pos(3 * kTokens);
        std::int32_t *d_src = nullptr, *d_dst = nullptr;
        cudaMalloc(&d_src, src_pos.size() * 4);
        cudaMalloc(&d_dst, dst_pos.size() * 4);
        for (int s = 0; s < kEvict; ++s) {
            const int old_slot = kEvict - 1 - s;  // device slot s holds host slot (3-s)
            const int new_slot = s;
            for (int t = 0; t < kTokens; ++t) {
                for (int a = 0; a < 3; ++a) {
                    src_pos[a * kTokens + t] = 100000 + old_slot * kTokens + t;
                    dst_pos[a * kTokens + t] = new_slot * kTokens + t;
                }
            }
            cudaMemcpy(d_src, src_pos.data(), src_pos.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dst, dst_pos.data(), dst_pos.size() * 4, cudaMemcpyHostToDevice);
            for (int l = 0; l < kLayers; ++l) {
                auto* kc = static_cast<std::int8_t*>(planes[static_cast<std::size_t>(l * 4)].data) +
                           fresh_index[s] * plane_strides[static_cast<std::size_t>(l * 4)];
                auto* ks = reinterpret_cast<__half*>(
                    planes[static_cast<std::size_t>(l * 4 + 2)].data) +
                           fresh_index[s] * plane_strides[static_cast<std::size_t>(l * 4 + 2)];
                kvmem_rerope_int8_g64_plane<kHeads>(kc, ks, 0, d_src, d_dst, 0);
                cudaStreamSynchronize(0);
                std::printf("post rerope slot %d layer %d err=%s\n", s, l,
                            cudaGetErrorString(cudaGetLastError()));
            }
        }
        cudaStreamSynchronize(0);
        std::printf("re-phased 4 blocks\n");

        // ---- verification (all via D2H) ----
        auto download_page = [&](int plane, int phys_page) {
            const int leading = geometry.planes[static_cast<std::size_t>(plane)].leading_extent;
            const int elem = geometry.planes[static_cast<std::size_t>(plane)].dtype ==
                                     DType::FP16
                                 ? 2
                                 : 1;
            std::vector<std::uint8_t> buf(
                static_cast<std::size_t>(elems_per_page(leading) * elem));
            CHECK(cudaMemcpy(buf.data(),
                             static_cast<std::uint8_t*>(planes[static_cast<std::size_t>(plane)].data) +
                                 phys_page * plane_strides[static_cast<std::size_t>(plane)] * elem,
                             buf.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "D2H plane %d", plane);
            return buf;
        };
        auto host_plane_page = [&](int plane, int slot) {
            const int leading = geometry.planes[static_cast<std::size_t>(plane)].leading_extent;
            const int elem = geometry.planes[static_cast<std::size_t>(plane)].dtype ==
                                     DType::FP16
                                 ? 2
                                 : 1;
            const std::size_t off = static_cast<std::size_t>(elems_per_page(leading) * elem * slot);
            const std::size_t count = static_cast<std::size_t>(elems_per_page(leading) * elem);
            return std::vector<std::uint8_t>(
                host_plane[static_cast<std::size_t>(plane)].begin() + off,
                host_plane[static_cast<std::size_t>(plane)].begin() + off + count);
        };

        // 1) Staged pages: position-free data bitwise vs the pre-evict host content.
        // (V planes only: the full-row H256 re-phase rewrites ALL K groups, unlike the
        // stale per-group-H64 model where groups 1..3 passed through untouched.)
        for (int s = 0; s < kEvict; ++s) {
            const int host_slot = kEvict - 1 - s;
            for (int l = 0; l < kLayers; ++l) {
                const int vc = l * 4 + 1;
                const int vs = l * 4 + 3;
                const auto vcd = download_page(vc, fresh_index[s]);
                const auto want_vc = host_plane_page(vc, host_slot);
                CHECK(vcd == want_vc, "V-codes slot %d bitwise", s);
                const auto vsd = download_page(vs, fresh_index[s]);
                const auto want_vs = host_plane_page(vs, host_slot);
                CHECK(vsd == want_vs, "V-scale slot %d bitwise", s);
            }
        }
        std::printf("position-free data bitwise ok\n");

        // 2) K full row of staged pages vs CPU model at the NEW slot (all 4 groups:
        // the full-row H256 re-phase rewrites every group).
        double worst = 0.0;
        std::vector<float> group(64);
        for (int s = 0; s < kEvict; ++s) {
            const int old_slot = kEvict - 1 - s;
            const int new_slot = s;
            for (int l = 0; l < kLayers; ++l) {
                const auto kcd = download_page(l * 4, fresh_index[s]);
                const auto ksd = download_page(l * 4 + 2, fresh_index[s]);
                for (int h = 0; h < kHeads; ++h) {
                    for (int t = 0; t < kTokens; ++t) {
                        const std::size_t midx =
                            (static_cast<std::size_t>(old_slot) * kLayers + l) * kHeads * kTokens +
                            static_cast<std::size_t>(h) * kTokens + t;
                        const KRow& stored = model[midx];
                        // Dequant the full row with the model's stored scales, then run the
                        // kernel's float sequence on the flat dim-indexed array.
                        std::vector<float> row(256);
                        for (int d = 0; d < 256; ++d) {
                            const float sg = __half2float(
                                __ushort_as_half(stored.scale_bits[d / 64]));
                            row[d] = static_cast<float>(stored.codes[d]) * sg;
                        }
                        hadamard256_host(row.data());
                        const std::int32_t pos_src[3] = {100000 + old_slot * kTokens + t,
                                                         100000 + old_slot * kTokens + t,
                                                         100000 + old_slot * kTokens + t};
                        const std::int32_t pos_dst[3] = {new_slot * kTokens + t,
                                                         new_slot * kTokens + t,
                                                         new_slot * kTokens + t};
                        rope_group0(row.data(), pos_src, true);
                        rope_group0(row.data(), pos_dst, false);
                        hadamard256_host(row.data());
                        for (int g = 0; g < kGroups; ++g) {
                            for (int d = 0; d < 64; ++d) { group[d] = row[g * 64 + d]; }
                            float absmax = 0.0f;
                            for (int d = 0; d < 64; ++d) {
                                absmax = std::max(absmax, std::fabs(group[d]));
                            }
                            const unsigned short ref_bits = __half_as_ushort(
                                __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f));
                            const float ref_scale = __half2float(__ushort_as_half(ref_bits));
                            // FP16 plane: element index in_page_elems(...) -> byte offset x2.
                            const std::size_t scale_byte = static_cast<std::size_t>(
                                                              in_page_elems(4, h, t, g)) *
                                                          2;
                            const float got_scale = __half2float(__ushort_as_half(
                                *reinterpret_cast<const unsigned short*>(&ksd[scale_byte])));
                            CHECK(std::fabs(got_scale - ref_scale) <= 1e-3f * ref_scale + 1e-7f,
                                  "scale p%d l%d h%d t%d g%d", s, l, h, t, g);
                            for (int d = 0; d < 64; ++d) {
                                // kcd is a byte buffer: codes are SIGNED int8.
                                const float got =
                                    static_cast<float>(static_cast<std::int8_t>(
                                        kcd[static_cast<std::size_t>(
                                            in_page_elems(256, h, t, g * 64 + d))])) *
                                    got_scale;
                                const double diff =
                                    std::fabs(static_cast<double>(got) - group[d]);
                                worst = std::max(worst, diff);
                                if (diff > 1.01 * static_cast<double>(got_scale) &&
                                    s == 0 && l == 0 && h == 0 && t == 0 && d < 8) {
                                    std::printf("detail g%d d=%d got=%f want=%f scale=%f\n", g, d,
                                                got, group[d], got_scale);
                                }
                                CHECK(diff <= 1.01 * static_cast<double>(got_scale),
                                      "code p%d l%d h%d t%d g%d d%d", s, l, h, t, g, d);
                                if (failures > 20) {
                                    std::printf("too many failures, abort\n");
                                    return 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        std::printf("worst re-baked dequant diff = %.6f\n", worst);

        // 3) Untouched pages 4..5 bitwise (all planes).
        for (int i = 4; i < kPages; ++i) {
            for (int p = 0; p < kTotalPlanes; ++p) {
                const auto dev = download_page(p, page_index[i]);
                const auto want = host_plane_page(p, i);
                CHECK(dev == want, "untouched plane %d page %d", p, i);
            }
        }
        std::printf("untouched pages bitwise ok\n");

        std::printf("failures=%d  VERDICT: %s\n", failures, failures == 0 ? "PASS" : "FAIL");
        cudaFree(backing);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
