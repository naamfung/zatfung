# zatfung — 疾风 (zat⁶ fung¹)

**zatfung**（粤语 *zat⁶ fung¹*，疾风）是一个 CUDA 优先的 C++ 推理引擎，  
面向 Bonsai 2 27B 的两种三元权重格式 —— 真三元 `PTQ1_0_G128`（base-3 三进制打包，  
28 B/128 权重）与假三元 `PQ2_0_G128`（2 bit 码，34 B/128），派生自 ZATFUNG 家族。

在 NINFER 架构下，本 fork **首发并独立实现了 KVMem** —— 一套将 KV cache 从显存上限里  
解耦出来的分层记忆机制（设计同源于 llama.cpp 侧的 laamaafung，实现按本引擎的分页  
KV / block_table 架构完全重写）：设备显存只驻留一个小的工作集窗口，完整上下文分层到  
宿主内存，于是两种三元打包的 27B 模型能以**极限方式跑进 8 GB 级显卡** —— 本机  
3060 Ti 实测 256K 逻辑上下文（机制与配置见下文 KVMem 专节）。

一个源码树同时覆盖四档 NVIDIA 架构：

| `CMAKE_CUDA_ARCHITECTURES` | 架构        | 代表显卡               | 说明                           |
| -------------------------- | --------- | ------------------ | ---------------------------- |
| `75`                       | Turing    | RTX 2080 Ti        | 最保守档：无 `cp.async`、无 bf16 MMA |
| `86`                       | Ampere    | RTX 3090 / 3060 Ti | **主要开发目标**（本机 3060 Ti）       |
| `89`                       | Ada       | RTX 4090 / 4070    | 本 fork 的原始目标线                |
| `120a`                     | Blackwell | RTX 5090           | 参考实现（NVFP4 W4A4 TMA）         |

四档都能全量构建，**"能编译"不等于"能跑"**，验收程度并不相同：

- `86`（本机 3060 Ti）已跑通构建 + 全量单测（111/111）+ 端到端生成。
- `75` 目前**只到编译层**：全量构建通过，MMA 与 warp 归约的 Turing 降级做过硬件级等价验证，  
  但尚无 2080 Ti 实机验收，且 prompt attention 的 tile 超出 Turing 每 block 64 KiB 的  
  opt-in 上限。缺口清单与后续步骤记录在内部设计文档中。

> **构建入口是 `builder.go`** —— 一个 Go 写的生产标准构建器。它会自己探测并拼装  
> MSVC 环境（不依赖 `vcvars64.bat`），因此在 `cmd.exe` 被禁用、Visual Studio  
> 生成器找不到 `cl.exe` 的环境里依然能从零构建。

---

### 克隆指南

> **项目声明（2026-10-03）**：本项目已转为内部项目，后续加强版不再开源发布。  
> **`vD` 是唯一公开的开源版本**；`vE` 已从远端删除，`vF` 及之后最新的 `vP` 分支为  
> 内部加强版。作为多个开源项目的作者，却由于近日的经历感觉开源有时真的无法保护自己，请容许我暂不考虑继续开源本项目，如须加强版请通过哔站联络作者我的账号「追逐南风」，或直接使用我公布的预编译版。外部用户请克隆 `vD` 通过编程代理自行适配；`vP` 仅限内部访问。

版本分支自 `master` 依次演进：`vA` → `vB` → `vC` → `vD` → `vE` → `vF`  
（vA–vC 为 KVMem 演进快照，vD 为前一验证线）。

- **克隆公开开源版（vD，外部用户入口）**：
  ```sh
  git clone -b vD https://github.com/naamfung/zatfung.git
  ```
- **克隆内部开发线（vP，仅限内部访问）**：全量测试套件 111/111 全绿（sm_86 实测）：
  ```sh
  如须加强版请通过哔站联络作者我的账号「追逐南风」，或直接使用我公布的预编译版……
  ```

---

### 编译指南

**工具链依赖**：Go 编译器（编译 builder 本身）、Visual Studio（MSVC C/C++ + Windows SDK）、  
CUDA Toolkit 12.x（nvcc）、Ninja；ccache 可选。

**基本用法**（在仓库根目录执行）：

```sh
go build -o builder.exe builder.go     # 得到自包含的构建器
./builder.exe -list                    # 环境自检：GPU / CUDA / MSVC / SDK / Ninja / ccache
./builder.exe                          # 探测本机 GPU 架构并完整构建
./builder.exe -arch 75 -j 16           # 显式指定架构
./builder.exe -fresh                   # 全量重建；-keep 只重置 CMake 状态
```

builder 内置了旧脚本踩过的全部坑的处理，最关键的几条：

- **不依赖 `vcvars64.bat`**：直接探测 VS 安装 / MSVC 工具集 / Windows SDK 版本，  
  手工拼出等价的 `INCLUDE` / `LIB` / `PATH`，让 Ninja 生成器拿到可用的 `cl.exe`。
- **默认 Ninja，不用 VS 生成器**（换回 VS 生成器用 `-gen vs`）。
- **ccache 只挂 CUDA**：包装本机本地化 MSVC 的 `cl.exe` 必崩，包装 nvcc 完全正常且能命中；  
  CUDA 实例正是耗时大头。
- **CUDA 版本逃生舱**：目标非 `120a` 时自动带上 `-DZATFUNG_ALLOW_LEGACY_CUDA=ON`，  
  75/86/89 三档用 CUDA 12.8 即可完整编译，不必为策略性版本号装新工具链。
- **代理变量剥离**：`HTTP_PROXY` 等会让 MSBuild 报 `MSB6001`，子进程环境强制剥离。

**常用参数**：`-arch 75|86|89|120a|native|auto`、`-j N`（并行度）、`-gen ninja|vs`、  
`-target T1,T2`（只构建指定目标）、`-media auto|on|off`（媒体解码，无 ffmpeg 时自动转 stub）、  
`-D "<CMake 参数>"`（额外 CMake 参数，**须自带 `-D` 前缀**，如 `-D="-DBUILD_TESTING=ON"`）、  
`-C <dir>`（指定仓库根，用于 worktree 跨分支构建）。

**构建目录与产物**：构建目录为 `build-SM<ARCH>`（多档可并存互不干扰），并在有  
git 的仓库里追加当前检出 —— 分支名（如 `build-SM86-vF`）或短哈希（detached 检出，  
如 `build-SM86-327372f`）—— 不同分支/提交的构建树因此天然分开，来回切换不必重建，  
配合共享 ccache 各自增量；无 `.git` 的分发包（GitHub ZIP）切不了分支，目录就是裸的  
`build-SM86`。`zatfung.exe` / `zatfung-serve.exe` / `zatfung-perplexity.exe` 连同  
`cudart64_*.dll`（以及存在时的 ffmpeg DLL）一起落位到构建目录根部，形成可直接运行  
的布局。每次构建结束自检产物齐全性，并自动清理构建目录树里不在落位路径上的过期同名副本  
（按三个产物名精确匹配，被占用的副本警告后留给下次构建）；编译日志见构建目录下的  
`builder-build.log` 与 `builder-configure.log`。

---

### 测试指南

带测试套件构建后用 ctest 运行（ctest 不由 builder 代跑）：

```sh
./builder.exe -arch 86 -D="-DBUILD_TESTING=ON"   # 带测试套件构建
cd build-SM86-vF
ctest --output-on-failure                        # 运行全部测试
```

- `vF` 全量 **111/111 通过**（sm_86，Windows 实测口径）。注册共 114 项：3 项 Disabled  
  （设计内，不计入统计分母），其余按架构能力 / 缺模型 Skip，无假失败。111 之外另有  
  3 项此前因 MSVC 编不过从未被 ctest 跑过（`1e9d57d` 起修复并跑绿）。
- **按架构能力自动跳过**：FP8 MMA（sm_89/120）与 NVFP4 W4A4（仅 sm_120）专属测试臂  
  在低档架构上以 `SKIP(77)` 跳过；A16 反量化臂任何架构恒编译、恒运行。  
  能力宏（`ZATFUNG_HAS_FP8_MMA` / `ZATFUNG_HAS_NVFP4_MMA`）由 CMakeLists 按 tier 派生。
- **real 系列测试**缺模型文件时 Skip；**媒体解码测试**仅在源码树提供  
  `ffmpeg/{include,lib}` 时注册（`ZATFUNG_BUILD_MEDIA_ACQUIRE`，builder 缺 ffmpeg 时自动转 stub）。
- 未运行的项目全部设计内（缺模型 / 缺工件 / 能力门控 / Disabled），无假失败。

---

### 运行指南

构建目录根部即可直接运行：

```sh
./build-SM86-vP/apps/zatfung-serve <model>.ninfer \
--host 127.0.0.1 --port 8008 --model-id Agentic-Model \
--max-context 262144 --kv-dtype k6v4 --kvmem --kvmem-budget 16384 --kvmem-gen-reserve 8192 \
--kv-device-tokens 24576 --max-concurrency 1 --zatcuk --no-thinking
```

- 在有多块 GPU 的机器上，用 `CUDA_VISIBLE_DEVICES` 指定序号（CUDA 的枚举顺序  
  未必与 `nvidia-smi` 一致）。`--max-context` / `--kv-capacity` / `--max-concurrency`  
  按显存与负载自行调。
- 运行期 KV 量化（`--kv-dtype`）不改权重，公开的 `.ninfer` 产物可直接使用。
- 完整选项契约、CLI 用法、性能与 PPL 评测记录在内部文档中（不随仓库发布）。

**长上下文与显存受限场景**：用下文的 KVMem —— 它把 KV cache 与显存上限解耦，  
是本 fork 在 ZATFUNG 架构下首发并独立实现的机制，真假三元 27B 都适用。

**Windows 提示**：存在 `--wddm-evictable-budget`（WDDM 驻留锁 + D3D12 共享堆），  
但它要求 GPU 不驱动桌面、仅 Windows 生效且不缩减权重占用，**不建议使用** ——  
长上下文的平台无关解法是下文的 KVMem。

---

### KVMem：把真假三元 27B 跑进 8 GB 显存

KVMem（仅 `zatfung-serve`）解决一个具体的死结：不开它时，KV cache 必须按整个上下文  
驻留显存 —— 按实测口径（int8 KV 约 127 KiB/token，含 KV 平面之外的分配）外推，  
256K 上下文约需 32.6 GiB 显存，8 GB 卡不可能；假三元 PQ2_0 的文本权重又比真三元  
大 1.17 GiB（7.03 GiB vs 5.86 GiB），显存余量更小。

它的做法是把「逻辑上下文」与「显存驻留」**解耦**：设备上只保留一个小页池作为  
常驻工作集，完整上下文的 KV 分层到宿主内存，窗口按需组装、换入换出 ——  
`--max-context` 因此只是逻辑上限，显存占用由池决定。四步机制（每步独立落地、带单测）：

1. **窗口组装（纯宿主决策层）** —— `sink`（头部频段，抗注意力稀释）与 `recent`  
   （尾部，保生成连续）恒驻留；其余配额按检索 / 注意力双信号选块，选中块升序  
   **紧凑打包**成窗口，经 block_table 发布给注意力内核。槽位即位置，紧凑打包同时  
   减少了内核迭代量。
2. **K 相位重铸（小 GPU 内核）** —— 打包改变了每个块的槽位，K 的 RoPE 相位必须  
   重铸：驻留块原地 re-RoPE，冷块从 raw-K 镜像重铸；remap 次数 / 位移双阈值触发  
   raw 重刷新，对抗 fp16 累积漂移。
3. **prefill 分块映射** —— prompt 不一次映射入池，按块映射；某块末端将超出池时，  
   先把窗口外的页停到宿主再继续。
4. **生成期回收** —— 输出把池跑满时按窗口换出、腾出 `--kvmem-gen-reserve` 的余量，  
   解码继续，不必为整个生成长度预留显存。

代价与约束：KV 平面的重相与块评分走同一族码（`int8-group64` / `i4` / `k6v4`，见下）；  
被回收的血脉不再命中兼容前缀缓存，下一轮重 prefill 而不是命中。

**KV 存储格式在本架构（sm_86）的可用性**：`bf16`、`int8`、`i4`、`k6v4` 四种 ——  
KVMem 在它们中只接受 `int8` / `i4` / `k6v4`（它的重相与块评分内核按该码族的平面布局  
寻址，别的存储会被当 int8 解出无意义评分，启动即拒绝）。`k8v4`（fp8 key + NVFP4  
value）与 `nvfp4` 需要 SM120 的 Blackwell 内核、`fp8` 与 `rk4v4` 系需要 sm_89 —— 在  
sm_86 上启动即报错并给出可用选项；`k8v4` 的 256K 全上下文实测数字（15.36 GiB KV）  
来自 16 GB 级 sm_89/sm_120 平台，与本机无关。日常配置用 `k6v4`（比 `int8` 再省 35%  
KV 显存），实测见下方「开源版 vD 与内部版 vK 的差异」。

本机实测（3060 Ti 8 GB）：

| 配置                                                                                                          | 结果                                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| PTQ1_0，`--max-context 262144 --kv-device-tokens 10752 --kvmem --kvmem-budget 8192 --kvmem-gen-reserve 2048` | 256K 逻辑上下文落在 168 页设备池上，余 500 MiB；prefill 354 tok/s（1,459-token prompt，`--no-thinking`，分块 2048）—— 该预填充数字取自内部版 vK；开源版 vD 同口径实测约 39 tok/s，见下节      |
| PTQ1_0，池小于 prompt（`--kv-device-tokens 640`：10 页 vs prompt 需要的 21 页）                                         | 三轮全部正确服务                                                                                                                                        |
| **PQ2_0 假三元、仅文本极限配置**（无视觉 / MTP / DFlash，见下方命令）                                                             | **8 GB 启动成功并通过真实生成**：规划后 free 仅 22.0 MiB，整机显存 7612 MiB（含桌面 ~375 MiB）；短 prompt 生成 27.1 tok/s（21-token prompt、60-token 输出 ×3，CUDA Graph 开启口径，详下节） |

KVMem 工作在 KV 侧、与权重打包格式正交：PTQ1_0（真三元）与 PQ2_0（假三元）的  
公开 `.ninfer` 产物都直接可用，无需重新转换。

配置启动（256K 极限配置，与上表 PTQ1_0 行同口径）：

```sh
./zatfung-serve.exe <PrismML-PTQ1_0.model>.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-dtype int8 \
  --kvmem --kv-device-tokens 10752 \
  --kvmem-budget 8192 --kvmem-gen-reserve 2048
```

**PQ2_0 假三元在 8 GB 卡上的极限配方**（不传 `--spec` / `--vision` 即不加载  
MTP / DFlash / 视觉权重，仅文本）：

```sh
./zatfung-serve.exe <PrismML-PQ2_0.model>.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-dtype int8 \
  --kvmem --kvmem-budget 1024 --kv-device-tokens 2048 \
  --kvmem-gen-reserve 256 --prefill-chunk 256 \
  --max-concurrency 1 --device-state-slots 0
```

每个旋钮都在换显存：窗口与池压到最小（1024/2048 token）、`--prefill-chunk 256`  
压激活、单并发零额外状态槽（每个额外的设备状态快照槽要 ~147 MiB，默认留一份，  
删掉即省回这一份）、CUDA Graph 保持默认开启 —— 它只要 12 MiB，  
换来 decode 约 +5%；显存极紧到 warmup 都过不去时才考虑 `--no-cuda-graph`。  
代价是吞吐与并发——这是"装得下"与"跑得动"之间的取舍。

**vD / vE 同口径对比**（同一台 3060 Ti、同一份 PQ2_0 仅文本极限配置、CUDA Graph  
开启、确定性请求（温度 0））：

| 口径                    | vD                       | vE                       |
| --------------------- | ------------------------ | ------------------------ |
| decode，60 token ×3 次  | 26.9 / 26.8 / 27.0 tok/s | 27.1 / 27.1 / 27.1 tok/s |
| decode，300 token ×2 次 | 26.8 / 26.6 tok/s        | 26.8 / 26.7 tok/s        |
| prefill，1,459 token   | 375 tok/s                | 375 tok/s                |

两版逐项打平（差异 ≤1%，测量噪声以内）：vE 相对 vD 无性能退化。

---

### 开源版 vD 与内部版（vK/vL）的差异

`vD` 是持续维护、公开可用的开源版本；`vK`/`vL` 是内部版本（vP 最新）。两版共用同一套引擎、  
同一份 `.ninfer` 产物与同一套命令行，差别在**能力边界与实测表现**。

本节数据来自同一台 3060 Ti 8 GB、同一份真三元 27B 产物（`bonsai2_27b_ptq1.ninfer`，权重 5.52 GiB）。  
两个口径：**口径 A**（4K 上下文、int8、提示 1,351 token）对比两版的基础能力差；**口径 B**（256K 上下文、  
k6v4、KVMem、内建排序题提示）是内部版 vL 在生产配置上的实测。只改分支与参数，别的都不动。

#### 一、结论

- **预填充快 7–11 倍**：同一提示的首字延迟从 34.9 s 降到 4.9 s（`--prefill-chunk 2048` 时  
  34.7 s → 3.5 s）。
- **投机解码的收益从 0.9–1.2× 变成 3.6–5.4×**：vD 开 MTP 基本白开，vK 每一档草稿窗都是  
  实打实的加速。
- **长上下文可以带投机一起跑**：同一配置 vD 启动即失败；内部版在 256K + KVMem 上实测  
  60.7–73.5 tok/s（口径 B）与 107.8 tok/s（口径 A）。
- 纯文本解码（23.4 对 23.6 tok/s）与显存规划（4K 480.2 MiB；256K 686.5 MiB / 164 页）  
  **两版逐项相同**。

差距不在权重、不在显存、也不在模型，而在 vK 对真三元（PTQ1_0）路径做的性能优化。

#### 二、能力差异

| 维度            | vD（开源）                                                      | vK / vL（商业最新）                                |
| ------------- | ----------------------------------------------------------- | -------------------------------------------- |
| 真三元（PTQ1_0）路径 | 基准实现                                                        | **性能优化版**                                    |
| MTP 草稿窗上限     | 5                                                           | **7**                                        |
| 全局调优总控        | 无（各开关独立，逐个手工调）                                              | **ZATCUK**（`--zatcuk`）：一处接管、现场自适应            |
| 长上下文 + 投机解码   | 装不下，启动即失败                                                   | **可用**（256K + KVMem 实测 60.7–73.5 tok/s，口径 B） |
| KV 精度         | `bf16` / `int8` / `fp8` / `nvfp4` / `k8v4` / `rk4v4`(-`e8`) | 上述 + **`i4`**、**`k6v4`**（约 1.53× int8 容量）    |
| KV 精度入口       | `--kv-dtype` 加一对 `--cache-type-k/-v`                        | 统一为 `--kv-dtype` 单一入口                        |


`--zatcuk` 就是 **ZATCUK** —— vK 的全局调优总控，不是某个开关的别名。它今天已接管的杠杆：  
投机解码的草稿窗与窗口旁边的显存分配（逐请求自适应）、KV 池容量与 KV 层开关（装不下时现场  
改配并逐项告知）、prefill 分块（省略 `--prefill-chunk` 时由它按显存与并发定价；显式给出的值  
仍作硬上限尊重）。设计上它是一处总控，DFlash2 等后续杠杆也会纳入同一处，用户不必逐个手调。

它需要上下文缓存，所以与 `--no-context-cache`（原名 `--no-prefix-reuse`，已显式拒绝并指路）  
互斥，同时给出报错。命令行形态下 `zatfung` 一次只跑一个请求、不保留跨请求状态，因此**单文件  
CLI 不支持 `--zatcuk`**：要固定草稿窗请用 `--spec mtp --draft-tokens N`，要 `--zatcuk` 请走  
`zatfung-serve`。

#### 三、同口径实测 · 口径 A（4K 上下文 · int8，两版对比）

**预填充**（引擎级 `zatfung_bench -p 2048`，1 次 warmup + 3 次取均）：

|        | vD         | vK              |
| ------ | ---------- | --------------- |
| pp2048 | 38.4 tok/s | **416.1 tok/s** |

**端到端**（HTTP 单次请求，同一提示）：

| 场景                          | vD         | vK              |
| --------------------------- | ---------- | --------------- |
| 预填充 · 4K / 分块 256           | 38.8 tok/s | **278.7 tok/s** |
| 预填充 · 4K / 分块 2048          | 39.0 tok/s | **387.4 tok/s** |
| 预填充 · 256K + KVMem          | 39.3 tok/s | **278.4 tok/s** |
| 解码 · 纯文本 4K                 | 23.4 tok/s | 23.6 tok/s      |
| 解码 · MTP k=3                | 27.9 tok/s | **86.0 tok/s**  |
| 解码 · MTP k=5                | 21.4 tok/s | **109.2 tok/s** |
| 解码 · MTP k=7                | 不支持（窗上限 5） | **128.3 tok/s** |
| 解码 · `--zatcuk accept`      | 无此选项       | **128.9 tok/s** |
| 解码 · `--zatcuk speed`       | 无此选项       | 122.2 tok/s     |
| 解码 · 256K + KVMem           | 23.3 tok/s | 23.7 tok/s      |
| 解码 · 256K + KVMem + MTP k=5 | **启动失败**   | **107.8 tok/s** |

首字延迟（同一提示）：vD **34.4–35.1 s**，vK **3.5–5.0 s**。开投机后 vK 的 reservation 为  
791.0 MiB（不开投机时 686.5 MiB，与 vD 相同）。

单次请求的 tok/s 会随生成内容抖动（同一配置下可差 ±20%），所以上表读的是量级：4–10× 的  
差距远在抖动之外；只有两版相等的那几格需要按「相同」理解。

上表所有数字都用**默认配置**跑出，不设任何环境变量。

#### 三·B、同口径实测 · 口径 B（256K 上下文 · k6v4 · KVMem，内部版 vL 实测）

生产长上下文配置：`--max-context 262144 --kv-dtype k6v4 --kvmem --max-concurrency 1`，  
KVMem 档位 =「窗:生成」（16:8 即 `--kvmem-budget 16384 --kvmem-gen-reserve 8192`），  
HTTP 单请求、`max_tokens` 等于该档生成量、固定温度 0.6 + seed。decode tok/s：

| 档位                     | `--zatcuk`（MTP 开） | 不带 `--zatcuk`（无 MTP）    |
| ---------------------- | ----------------- | ----------------------- |
| 16:8 · 生成 8K · 关思考     | **73.5 tok/s**    | 22.5 tok/s              |
| 16:8 · 生成 8K · 开思考     | **60.7 tok/s**    | 21.9 tok/s              |
| 32:8 · 生成 8K · 关/开思考   | 22.7 / 22.2 tok/s | 22.4 / 22.1 tok/s       |
| 32:16 · 生成 16K · 关/开思考 | 22.3 / 22.0 tok/s | 8 GB 上需 `--zatcuk` 才能启动 |

读法：

- **投机解码在长上下文上依旧 2.7–3.3×**：同窗同长度，MTP 开 60.7–73.5、关 21.9–22.5。
- **窗长是 decode 的第一杠杆**：16K 窗 60–73 tok/s、32K 窗 ~22 tok/s——`--kvmem-budget`  
  每小一半，decode 近 3×；选档就是选「常驻多少、跑多快」。
- **32:16 档在 8 GB 上需搭配 `--zatcuk`**：无它时该档装不下（启动即报错并写明缺口）。

#### 四、投机解码的收益

同一份权重、同一个草稿窗，两版的收益差一个量级：

| 解码 tok/s | 纯文本  | MTP k=3     | MTP k=5      |
| -------- | ---- | ----------- | ------------ |
| vD       | 23.4 | 27.9（1.19×） | 21.4（0.91×）  |
| vK       | 23.6 | 86.0（3.64×） | 109.2（4.63×） |

vD 把草稿窗开大反而更慢（k=5 低于不开），vK 每一档都是实打实的加速；窗上限也不同  
（7 对 5）。256K 那一行更直接：同一配置 vD 起不来，vK 有 107.8 tok/s。

#### 五、复现

同一份产物、同一台机器，只改分支与参数：

```sh
# 引擎级预填充
zatfung_bench.exe <PTQ1_0>.ninfer -p 2048 --kv-dtype int8 --max-ctx 4096 -r 3 --warmup 1

# 端到端
zatfung-serve.exe <PTQ1_0>.ninfer \
  --max-context 4096 --kv-dtype int8 --prefill-chunk 256 \
  --max-concurrency 1 --no-thinking \
  --spec mtp --draft-tokens 5      # vK/vL 上也可换成 --zatcuk accept

# 长上下文（口径 B）
zatfung-serve.exe <PTQ1_0>.ninfer \
  --max-context 262144 --kv-dtype k6v4 \
  --kvmem --kvmem-budget 16384 --kvmem-gen-reserve 8192 \
  --max-concurrency 1 --zatcuk
```

> 口径 A 的两版数据由同一驱动程序逐格落盘；口径 B 的数字出自矩阵 harness `testser`  
> （逐格完整命令原文与请求体随结果归档）。

---

### 模型下载（template-fetch）

`tools/template-fetch` 从 HuggingFace **只下载 3.11 GiB**，自建 `pack_zatfung.py`  
所需的 19.03 GiB 模板 artifact —— 模板里 771 个 `text/*` 对象由打包器自己生成、从不读取，  
真正需要的 419 个对象（vision 333 + dflash2 66 + mtp 12 + frontend 6 + text/draft_head 2）  
恰好集中在文件两端，用 HTTP Range 取两段即可（省 83.6%）：

```sh
cd tools/template-fetch
go build -o template-fetch.exe main.go
./template-fetch.exe                              # 默认：下载并产出可直接使用的完整模板
./template-fetch.exe -out D:\tpl\qwen3_8_27b_v2.ninfer   # 指定输出路径
./template-fetch.exe -j 8                         # 并发块数（默认 4）
```

- **两种输出模式**：默认产出完整模板（可直接使用，中间 15.92 GiB 区段留零）；  
  `-slim` 只产出 3.11 GiB 的 extract（需自行还原）。
- **可靠性**：显式要求 `206 Partial Content`（服务端忽略 Range 即报错）；  
  断点续传（`<out>.tplfetch.json`，中断重跑接着下）；全量 SHA-256 校验。
- **revision 固定**：URL 钉在 `dc370fb6295a`（v2 容器）。HF 上的 `main` 已换成 v3  
  artifact，与本引擎的 v2 读取器不兼容 —— 程序会 HEAD 比对 `Content-Length`，  
  源文件被替换时直接报错而不是产出坏模板。

---

### 模型转换（ternary-convert）

`tools/ternary-convert` 把 **Ternary-Bonsai-2-27B** 的 GGUF（PQ2_0 / PTQ1_0 两种三元打包）  
转成 zatfung 能直接加载的 `.ninfer` artifact。主转换器是 `pack_zatfung.py`  
（由上游 `pack.py` 改造：路径环境变量化 + 修掉 PTQ1_0 自检 bug）：

```sh
# 前提：CPU 版 torch 即可，不需要 CUDA 版
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu

cd tools/ternary-convert
set ZATFUNG_ROOT=<zatfung 仓库根>              # 提供 tools.artifact
set ZATFUNG_TEMPLATE=<上一步产出的完整模板路径>
set ZATFUNG_GGUF=<GGUF 路径>
python pack_zatfung.py check                  # 先验证几何/解码/字节往返，不写文件
python pack_zatfung.py build <输出>.ninfer    # 产出
```

- **GGUF 来源**：`Ternary-Bonsai-2-27B-PQ2_0.gguf`（type 142）与  
  `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（type 143），需自行从发布方获取。  
  **权重版权归其原作者与发布方所有，本仓库不包含、也不重新分发任何模型权重**；  
  由权重产出的 `.ninfer` 制品属权重派生品，再分发义务以权重原许可为准。
- **产物为何比 GGUF 大 ~3.1 GiB**：文本权重一分未变，膨胀全部来自 GGUF 里没有的  
  借用模块（dflash2 / mtp / draft_head / vision / frontend，共 419 对象 3.114 GiB，  
  全部从模板借用）。
- 完整操作记录与踩坑见 [`tools/ternary-convert/CONVERSION_NOTES.md`](tools/ternary-convert/CONVERSION_NOTES.md)。

---

> 感谢： **[Crane](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)**（ZATFUNG Ada 线的假三元 Bonsai 2 27B 移植）、**[Ninfer](https://github.com/Neroued/ninfer)** 原始项目或做相关移植工作的开源作者。
