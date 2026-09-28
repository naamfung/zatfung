# zatfung — 疾风 (zat⁶ fung¹)

**zatfung**（粤语 *zat⁶ fung¹*，疾风）是一个 CUDA 优先的 C++ 推理引擎，
面向 Bonsai 2 27B 的两种三元权重格式 —— 真三元 `PTQ1_0_G128`（base-3 三进制打包，
28 B/128 权重）与假三元 `PQ2_0_G128`（2 bit 码，34 B/128），派生自 NINFER 家族。

在 NINFER 架构下，本 fork **首发并独立实现了 KVMem** —— 一套将 KV cache 从显存上限里
解耦出来的分层记忆机制（设计同源于 llama.cpp 侧的 laamaafung，实现按本引擎的分页
KV / block_table 架构完全重写）：设备显存只驻留一个小的工作集窗口，完整上下文分层到
宿主内存，于是两种三元打包的 27B 模型能以**极限方式跑进 8 GB 级显卡** —— 本机
3060 Ti 实测 256K 逻辑上下文（机制与配置见下文 KVMem 专节）。

一个源码树同时覆盖四档 NVIDIA 架构：

| `CMAKE_CUDA_ARCHITECTURES` | 架构 | 代表显卡 | 说明 |
|---|---|---|---|
| `75` | Turing | RTX 2080 Ti | 最保守档：无 `cp.async`、无 bf16 MMA |
| `86` | Ampere | RTX 3090 / 3060 Ti | **主要开发目标**（本机 3060 Ti） |
| `89` | Ada | RTX 4090 / 4070 | 本 fork 的原始目标线 |
| `120a` | Blackwell | RTX 5090 | 参考实现（NVFP4 W4A4 TMA） |

四档都能全量构建，但**"能编译"不等于"能跑"**，验收程度并不相同：

- `86`（本机 3060 Ti）已跑通构建 + 全量单测（111/111）+ 端到端生成。
- `75` 目前**只到编译层**：全量构建通过，MMA 与 warp 归约的 Turing 降级做过硬件级等价验证，
  但尚无 2080 Ti 实机验收，且 prompt attention 的 tile 超出 Turing 每 block 64 KiB 的
  opt-in 上限。缺口清单与后续步骤见
  [`docs/zatfung-sm86-sm75-port.md`](docs/zatfung-sm86-sm75-port.md) 第 8 节。

> **构建入口是 `builder.go`** —— 一个 Go 写的生产标准构建器。它会自己探测并拼装
> MSVC 环境（不依赖 `vcvars64.bat`），因此在 `cmd.exe` 被禁用、Visual Studio
> 生成器找不到 `cl.exe` 的环境里依然能从零构建。

---

### 克隆指南

> **商业化声明（2026-09-28）**：本项目已转为商业项目，后续增强版本不再开源发布。
> **`vD` 是唯一持续维护并公开的开源版本**；`vE` 已从远端删除，`vF` 及之后的分支为
> 内部开发线、不再公开推送。外部用户请克隆 `vD`；`vF` 仅限内部访问。

版本分支自 `master` 依次演进：`vA` → `vB` → `vC` → `vD` → `vE` → `vF`
（vA–vC 为 KVMem 演进快照，vD 为前一验证线）。

- **克隆公开开源线（vD，外部用户入口）**：
  ```sh
  git clone -b vD https://github.com/naamfung/zatfung.git
  ```
- **克隆内部开发线（vF，仅限内部访问）**：全量测试套件 111/111 全绿（sm_86 实测），
  含 PTQ1_0 MMA prefill、解码诊断行等 vD 之后的能力：
  ```sh
  git clone -b vF <内部仓库地址>
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
- **CUDA 版本逃生舱**：目标非 `120a` 时自动带上 `-DNINFER_ALLOW_LEGACY_CUDA=ON`，
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
`build-SM86`。`ninfer.exe` / `ninfer-serve.exe` / `ninfer-perplexity.exe` 连同
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
  能力宏（`NINFER_HAS_FP8_MMA` / `NINFER_HAS_NVFP4_MMA`）由 CMakeLists 按 tier 派生。
- **real 系列测试**缺模型文件时 Skip；**媒体解码测试**仅在源码树提供
  `ffmpeg/{include,lib}` 时注册（`NINFER_BUILD_MEDIA_ACQUIRE`，builder 缺 ffmpeg 时自动转 stub）。
- 未运行的项目全部设计内（缺模型 / 缺工件 / 能力门控 / Disabled），无假失败。

---

### 运行指南

构建目录根部即可直接运行：

```sh
./ninfer-serve.exe <model>.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 131072 --kv-capacity auto --kv-dtype rk4v4-e8 \
  --max-concurrency 2 --device-state-slots 2 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --prefill-chunk 2048
```

- 在有多块 GPU 的机器上，用 `CUDA_VISIBLE_DEVICES` 指定序号（CUDA 的枚举顺序
  未必与 `nvidia-smi` 一致）。`--max-context` / `--kv-capacity` / `--max-concurrency`
  按显存与负载自行调。
- 运行期 KV 量化（`--kv-dtype`）不改权重，公开的 `.ninfer` 产物可直接使用。
- 完整选项契约见 [`docs/serving.md`](docs/serving.md)；CLI 用法见
  [`docs/cli.md`](docs/cli.md)；性能与 PPL 评测见
  [`docs/performance.md`](docs/performance.md) / [`docs/perplexity.md`](docs/perplexity.md)。

**长上下文与显存受限场景**：用下文的 KVMem —— 它把 KV cache 与显存上限解耦，
是本 fork 在 NINFER 架构下首发并独立实现的机制，真假三元 27B 都适用。

**Windows 提示**：存在 `--wddm-evictable-budget`（WDDM 驻留锁 + D3D12 共享堆），
但它要求 GPU 不驱动桌面、仅 Windows 生效且不缩减权重占用，**不建议使用** ——
长上下文的平台无关解法是下文的 KVMem。

---

### KVMem：把真假三元 27B 跑进 8 GB 显存

KVMem（仅 `ninfer-serve`）解决一个具体的死结：不开它时，KV cache 必须按整个上下文
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

代价与约束：KV 平面按 `int8-group64` 存储（重相与打分都经这个编解码器）；
被回收的血脉不再命中兼容前缀缓存，下一轮重 prefill 而不是命中。

本机实测（3060 Ti 8 GB）：

| 配置 | 结果 |
|---|---|
| PTQ1_0，`--max-context 262144 --kv-device-tokens 10752 --kvmem --kvmem-budget 8192 --kvmem-gen-reserve 2048` | 256K 逻辑上下文落在 168 页设备池上，余 500 MiB；prefill 354 tok/s（1,459-token prompt，`--no-thinking`，分块 2048） |
| PTQ1_0，池小于 prompt（`--kv-device-tokens 640`：10 页 vs prompt 需要的 21 页） | 三轮全部正确服务 |
| **PQ2_0 假三元、仅文本极限配置**（无视觉 / MTP / DFlash，见下方命令） | **8 GB 启动成功并通过真实生成**：规划后 free 仅 22.0 MiB，整机显存 7612 MiB（含桌面 ~375 MiB）；短 prompt 生成 27.1 tok/s（21-token prompt、60-token 输出 ×3，CUDA Graph 开启口径，详下文速查表） |

KVMem 工作在 KV 侧、与权重打包格式正交：PTQ1_0（真三元）与 PQ2_0（假三元）的
公开 `.ninfer` 产物都直接可用，无需重新转换。

配置启动（256K 极限配置，与上表 PTQ1_0 行同口径）：

```sh
./ninfer-serve.exe <PrismML-PTQ1_0.model>.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-dtype int8 \
  --kvmem --kv-device-tokens 10752 \
  --kvmem-budget 8192 --kvmem-gen-reserve 2048
```

**PQ2_0 假三元在 8 GB 卡上的极限配方**（不传 `--spec` / `--vision` 即不加载
MTP / DFlash / 视觉权重，仅文本）：

```sh
./ninfer-serve.exe <PrismML-PQ2_0.model>.ninfer \
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

| 口径 | vD | vE |
|---|---|---|
| decode，60 token ×3 次 | 26.9 / 26.8 / 27.0 tok/s | 27.1 / 27.1 / 27.1 tok/s |
| decode，300 token ×2 次 | 26.8 / 26.6 tok/s | 26.8 / 26.7 tok/s |
| prefill，1,459 token | 375 tok/s | 375 tok/s |

两版逐项打平（差异 ≤1%，测量噪声以内）：vE 相对 vD 无性能退化。


---

### 本机性能速查（vF，3060 Ti 8 GB）

CUDA Graph 开启，确定性请求（temperature 0）。首字延迟 = 提交请求到收到第一个 token
的时间。两种三元格式各按其 8 GB 配置实测，思维开/关两种口径：

**Q2 · PQ2_0 假三元，仅文本极限配置**（free 10 MiB，见上节配方）：

| 场景 | 输入 | 输出 | 首字延迟 | 填充速度 | 生成速度 |
|---|---|---|---|---|---|
| 短问答 | 20 tok | 60 tok | 0.74 s | 27 tok/s | 27.1 tok/s |
| 短文档摘要 | 511 tok | 63 tok | 1.40 s | 365 tok/s | 26.9 tok/s |
| 中文档摘要 | 1,003 tok | 65 tok | 2.66 s | 377 tok/s | 26.8 tok/s |
| 长文档摘要 | 1,459 tok | 74 tok | 3.89 s | 375 tok/s | 26.7 tok/s |
| 长输出创作 | 21 tok | 300 tok | 0.43 s | 49 tok/s | 26.7 tok/s |

**Q2 · 同配置，开思维（默认模板）**：

| 场景 | 输入 | 输出 | 首字延迟 | 填充速度 | 生成速度 |
|---|---|---|---|---|---|
| 短问答 | 60 tok | 60 tok | 1.26 s | 48 tok/s | 27.1 tok/s |
| 短文档摘要 | 551 tok | 128 tok | 1.90 s | 290 tok/s | 26.9 tok/s |
| 中文档摘要 | 1,043 tok | 128 tok | 2.88 s | 362 tok/s | 26.8 tok/s |
| 长文档摘要 | 1,499 tok | 125 tok | 3.87 s | 387 tok/s | 26.5 tok/s |
| 长输出创作 | 61 tok | 300 tok | 1.02 s | 60 tok/s | 26.7 tok/s |

**Q1 · PTQ1_0 真三元，宽配置**（free 744 MiB：更大的池与窗口、默认状态槽）：

| 场景 | 输入 | 输出 | 首字延迟 | 填充速度 | 生成速度 |
|---|---|---|---|---|---|
| 短问答 | 20 tok | 60 tok | 1.05 s | 19 tok/s | 23.3 tok/s |
| 短文档摘要 | 511 tok | 63 tok | 1.83 s | 280 tok/s | 23.2 tok/s |
| 中文档摘要 | 1,003 tok | 65 tok | 3.15 s | 318 tok/s | 23.2 tok/s |
| 长文档摘要 | 1,459 tok | 72 tok | 4.51 s | 324 tok/s | 23.0 tok/s |
| 长输出创作 | 21 tok | 300 tok | 0.66 s | 32 tok/s | 23.1 tok/s |

**Q1 · 同配置，开思维（默认模板）**：

| 场景 | 输入 | 输出 | 首字延迟 | 填充速度 | 生成速度 |
|---|---|---|---|---|---|
| 短问答 | 60 tok | 60 tok | 1.85 s | 32 tok/s | 23.1 tok/s |
| 短文档摘要 | 551 tok | 128 tok | 2.56 s | 215 tok/s | 23.1 tok/s |
| 中文档摘要 | 1,043 tok | 123 tok | 3.51 s | 297 tok/s | 22.9 tok/s |
| 长文档摘要 | 1,499 tok | 128 tok | 4.54 s | 330 tok/s | 22.8 tok/s |
| 长输出创作 | 61 tok | 300 tok | 1.59 s | 38 tok/s | 22.8 tok/s |

- **思维模板的行为**：开思维后 chat 模板在 prompt 前后追加约 40 token（上表输入列
  含模板），首字延迟随之略增；每 token 的生成速率与关思维完全一致 —— 思维只是
  把预算花在思考 token 上，不改变 decode 内核的吞吐。开启后输出几乎总是耗尽
  `max_tokens`（finish=length），需要完整答案时记得调大输出预算。
- 短 prompt 的填充速率含固定开销（`--prefill-chunk 256` 的固定开销在小 prompt 上
  占比更高），参考意义在长 prompt 梯度：0.5k → 1.5k 上 Q2 稳定在 362–387 tok/s、
  Q1 稳定在 215–330 tok/s。
- 生成速度两种格式、思维开/关都与输入输出长度基本无关：Q2 **26.5–27.1 tok/s**、
  Q1 **23.0–23.9 tok/s**（差约 14%，来自 base-3 解码的额外指令）。
- 对照说明：上游同模型在 RTX 4070 Ti SUPER（sm_89）记录 decode 100.8 t/s（MTP
  draft 3）—— 本机 8 GB 极限配置不加载 MTP/DFlash，且架构低两档，两口径不可直接比较。
- **Q1/Q2 权衡**：Q1 省 1.17 GiB 显存（free 744 MiB，可开更大的池与窗口），代价是
  填充 -15%、生成 -14%；Q2 把显存用到只剩 10 MiB 换全速。MMA prefill 路径两种格式
  都已具备（Q1 的实现见
  [`docs/ptq1-prefill-mma-design.md`](docs/ptq1-prefill-mma-design.md)）；Q1 的解码
  热路径还带一张每 CTA 的 base-3 数字查找表（值与算术解码逐位一致）。

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

更多文档见 [`docs/README.md`](docs/README.md) 索引。

---

> 本仓库是衍生作品，衍生自上游 **[CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)**（NINFER Ada 线的假三元 Bonsai 2 27B 移植）。
