# 映射族设计：三端能力阶梯（TileIR → Backend）

2026-10-23 定稿。本文档是 `sile-lsir/src/mapping.rs` 映射族与三端拼写的设计记录。pass 拓扑的规范裁决在 `docs/PIPELINE_SPEC.md`（冲突时以它为准），里程碑与验收线在 `docs/BACKEND_PLAN.md`，学术审计在 `docs/ACADEMIC_GROUNDING_AUDIT.md`。

## 1. 设计立场

映射族回答三个问题：每个 tile 值存在哪、哪个维度分给线程、哪里需要 barrier。它只改放置、不改语义（cutile 铁律）。backend 是方言拼写器。

参照系是 Triton 的三层式（TTIR 冻结 / TTGIR 布局与映射 / LLVM IR 标量优化）。本管线的对应物：

| Triton | sile | 措辞边界 |
|---|---|---|
| TTGIR blocked layout | `MappingPolicy`（每 target 全局常量）+ strip 循环 | 不是 per-value layout；布局是常量，元素归属由 `e % W` 循环改写表达 |
| reduction layout + shuffle/tree combine | reduce 阶段的 `ReducePolicy` | Triton 按 size 选 warp/shared 路径；本管线按 target 能力选 |
| dot-operand layout + shared staging | mma 阶段的 `MmaPolicy` | 同上 |
| mapped-loop unrolling、`sizePerThread` fragment | cleanup 阶段的 unroll/privatize | 同源 |

TileLang 只取「tile 原语独立于线程映射存在」这半面（`PIPELINE_SPEC.md` §2.1 既有裁决）：`TileReduce`/`TileMma` 活到映射层，schedule 概念不引入。

**不引入 per-value layout 系统**。理由：policy 全局常量模型已在 WGSL 端验证达标（softmax 96–103% / rms_norm 112–128% / gemm ≈100%）；完整布局推断（sizePerThread/threadsPerWarp/warpsPerCTA 三元组、convert_layout、swizzle）是路线图项（§7），等出现一个达标线以下、且 strip 模型无法表达的 kernel 再立项。

## 2. 三端能力矩阵与降级阶梯

每个 tile 级原语在每端有一条降级阶梯。policy 站在阶梯顶端；映射失败或形状不满足时逐级下落；任何一级拼不出来都 fail-closed（契约 verify 拒绝），绝不静默拼出错误代码。

| 原语 | Metal（优先 → 降级） | WGSL | C | PTX（M4） |
|---|---|---|---|---|
| TileReduce | **SimdGroupTree** → WorkgroupTree → Serial | WorkgroupTree → Serial | omp simd reduction | warp shuffle + shared 树 |
| TileMma | **SimdGroupTiles** → ThreadStrips → Serial | ThreadStrips → Serial | 寄存器分块 micro-kernel | mma.sync m16n8k8 |
| 同步 | `threadgroup_barrier`（全档可用） | `workgroupBarrier` | 无需 | `bar.sync` |
| tile 存储 | threadgroup 数组 / thread 私有 | `var<workgroup>` / 函数私有 | 栈数组 | `.shared` / 寄存器 |
| 向量化 | `float4` 装载（路线图） | — | `#pragma omp simd` | `v4.f32` |
| 整数宽度 | 合成循环一律 `int`（§4.1） | **i64→i32/u32 全端降级（§5）** | lowering 原样（i64） | M4 期定 |

能力依据：

- **simd-group 归约原语**（`simd_sum`/`simd_product`/`simd_min`/`simd_max`）：MSL 2.0 起全设备可用（iOS 11+/macOS 10.13+，Apple 与 Intel/AMD GPU 一体）。部分填充的 simd-group 按活跃 lane 归约并广播，实证见 §6.3。静态开启，不做运行期探测。
- **simdgroup_matrix**（8×8×8 乘累加）：**Apple7+ family 特性**（MTLGPUFamily::Apple7，即 A14/M1 起）。实证：AMD Radeon Pro 5500M（Mac2/Common3）上 `newLibrary` 通过、`newComputePipelineState` 失败（"SC compilation failure: There is a call to an undefined label"）——前端不拒绝，pipeline 才拒绝，故 capability 探测必须进 policy（`simdgroup_matrix_supported()` 以 Apple7 探测，非 Apple7 即留 ThreadStrips）。叠加形状要求 m,n,k 为 8 的倍数、操作数在 threadgroup 存储、元素类型 float/half；任一不满足即降级 ThreadStrips。
- **threadgroup 存储预算**：WGSL 规范保证 16KB（`shared_cap: 16384`）；MSL 保证 32KB。两端统一取保守值 16384，一个 kernel 的映射决策因此可以跨端复用同一形状判定。
- **整数宽度**：AIR 无原生 64 位整数指令（§4.1），wgpu 路径同样以其为终点，32 位是两端的共同快路径；WGSL 端 i64 的能力与开销记录见 §5。

## 3. pass 逐条理论依据

映射族五阶段 `policy → threads → reduce → mma → cleanup`（`MAPPING_STAGES`）。每阶段一条：决策、不变量、文献。

### 3.1 policy — 线程数与布局常量

**决策**：一 tile = 一 workgroup；线程数 = min(plumbing extent, tile_len, 256) 向下取 2 幂。

**依据**：
- 线程数取 2 幂：树归约按 stride 减半走直线代码（Blelloch, *Prefix Sums and Their Applications*, 1990），2 幂使每步无残组。
- 上限 256：barrier 一轮的代价随线程数增长，超出 tile 元素数的线程全程空转仍要参加每轮 barrier（Harris, *Optimizing Parallel Reduction in CUDA*, GTC 2007 的规模论证）。
- 一 tile = 一 workgroup：tile 是内核的计算与访存单元（Tillon et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*, MAPL 2019 的 blocked 编程模型），workgroup 是 barrier 与共享存储的分配单元，两者对齐使共享存储恰好装下一个 tile 的协作面。
- tile 内元素到线程的分配 `e = tid + k·W`（cyclic）：线程间步长 W 使相邻线程访问相邻元素，折叠成合并访存；同一线程的元素相距 W·stride，构成寄存器分块的行（Goto & Geijn, *Anatomy of High-Performance Matrix Multiplication*, TOPLAS 2008 的 register tiling）。与 Triton 的 `sizePerThread`（每线程连续元素）不同，此处是 cyclic 交错——偏离已记录（audit「threads 失真」条），strip 循环的形态决定了 cyclic 是循环改写能表达的最小方案。

**不变量**：每循环每元素恰好一次写（`check_strip_ownership`）。线程数是 `Policy` 数据，五阶段只读。

### 3.2 threads — strip 改写

**决策**：lowering 的 per-element 循环（plumbing 前缀命名）从 `init=0, step=1` 改写为 `init=tid, step=W`。

**依据**：thread coarsening（Magni, Dubach & O'Boyle, PACT 2014：把线程级并行收进线程内循环以换取寄存器复用与合并访存）的对偶——这里反过来，把循环级并行摊到线程。所有权判定是 owner-computes 规则（Wolfe, *High Performance Compilers for Parallel Computing*, 1996）的离散形：元素 `e` 的计算分给线程 `e % W`。

**不变量**：只重写循环边界，循环体内的依赖（跨迭代累加除外）不受影响；用户 `for` 不在改写面内。

### 3.3 reduce — TileReduce 的三档形态

**Serial**：intrinsic 原样留给方言拼串行循环。

**WorkgroupTree**：每线程 strip partial 写 shared scratch，barrier，log2(W) 步两两折叠（每步一个 guard combine + 一个 barrier），copy-out，尾 barrier。依据：树归约的 log 步深（Blelloch 1990）+ 两阶段结构（Harris 2007：intra-block 与 cross-block 分离）。内存序：scratch 写 → barrier → 读，是 GPU 内存模型的 release/acquire 配对（WGSL 规范 workgroupBarrier 语义、MSL threadgroup_barrier 内存旗标）。单位元是 monoid 单位元（Sum=0, Product=1, Max=-inf, Min=+inf），与 TileIR fold 携带的 identity 一致（cdf56ca 修复）。

**SimdGroupTree**（Metal 新增）：strip partial 先做 simd-group 内硬件归约（一条 `simd_*` 指令，log2(32)=5 步由硬件直线完成），随后只对 W/32 个 simd partial 走 shared 小树。依据：warp/shuffle 归约取代 shared 往返（Lindholm et al., *Programming the CUDA Architecture*, 2008 的 warp 原语；Harris 2007 的两阶段论证——第一阶段的共享往返是纯开销）。降级链：W≤32 单条 simd 归约，零 shared、零 barrier；W>32 补一轮 shared 小树。

**新不变量（simd-uniform 控制流）**：MSL 要求 simd 原语被 simd-group 的全部 lane 在一致控制流中遇到。映射生成的树步全部位于 guard `tid % 32 < stride` 内——stride 是常量、`tid % 32` 是 lane 在 simd 内的固定序号，故谓词对每个 simd-group 内一致，满足要求。此不变量写入 SimdGroupTree 阶段注释并由测试覆盖。

### 3.4 mma — TileMma 的三档形态

**Serial**：intrinsic 原样留给方言拼串行循环。

**ThreadStrips**：操作数 tile 翻入 shared 存储，barrier 交接（写 → barrier → 读），每线程按 `tid + k·W` 累加 accumulator strip。依据：shared 分块 + 寄存器累加是 blocked GEMM 的最小形态（Goto & Geijn 2008 的 outer-product 形式）；barrier 交接与 Triton `tl.dot` 的 shared staging 同构。

**SimdGroupTiles**（Metal 新增）：操作数同样翻入 threadgroup 并 barrier 交接；乘累加本身保留 `TileMma` intrinsic，由 Metal 方言拼成 8×8×8 simdgroup_matrix 序列：`simdgroup_matrix<float,8,8>` 累加块声明、`simdgroup_matrix_load`（B 侧经 transpose 参数吃行主序存储的列视图）、`simdgroup_multiply_accumulate` 链、`simdgroup_matrix_store` 归位。依据：MSL simdgroup matrix 的 8×8×8 fragment 与 PTX `mma.m8n8k8` 的 fragment 布局同族（每 lane 持定数个累加元素），是 Apple GPU 的 tensor 级原语。设备契约（Apple7+ family，运行期探测）与形状契约（m,n,k 为 8 的倍数，映射期检查）任一不满足即降 ThreadStrips。

**不变量**：操作数的其他读者不被破坏——翻入 shared 的 tile 必须只被该 mma 消费（使用面检查）；barrier 落在均匀汇合块（全线程到达同一 barrier 指令），满足 GPU 屏障的 uniformity 要求（WGSL/MSL 规范）。

### 3.5 cleanup — 放置兑现

unroll（trip 预算）、`expand_owner_strips`/`privatize_owner_tiles`（tile 收缩成线程 fragment）、`share_owner_tiles`（协作 tile 翻 threadgroup）、sroa、storage、DCE。

**依据**：fragment 化是 scalar replacement of aggregates（Cytron et al., *Efficiently Computing Static Single Assignment Form*, TOPLAS 1991 的存储视角；LLVM SROA 同算法族）——常数下标访问的数组退化为 SSA 值寄存器；unroll 服务于它（展开后下标才是常数）。寄存器 fragment 一次形成后不再有中转拷贝，是寄存器分配前的死存储消除（storage/DCE）。

**已知的 fitting 弱点**（audit 记录，本次路线）：`share_owner_tiles` 在无跨线程访问处也插 barrier（喂 `parallel_contract` 的存在性检查）；修复方向见 §6.2。

### 3.6 拼写与契约（emit 层）

backend 是方言拼写器：一个共享遍历（`emit.rs` 的 `TargetEmitter` + 结构化 CFG 发射）+ 每 target 一张拼写表（`TargetDialect`）。依据：方言分层（Lattner et al., *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*, CGO 2021 的 dialect separation）——下降与拼写分离使每端只承担「语法」而非「决策」。

fail-closed 契约三层：输入契约（lowering 不产的形态出现即拒绝，如 Barrier/ThreadId/SimdReduce）、映射后契约（`parallel_contract`：tile intrinsic 必须已物化、shared 与 barrier 配对）、端内验证（naga 解析 WGSL、Metal `newLibrary` 真编译、cc 真编译）。静默降级（映射失败回退 serial）只允许存在一次且有观测开关。

## 4. Metal 拼写要点

| LSIR 形态 | MSL 拼写 |
|---|---|
| `Intrinsic::Barrier{Workgroup}` | `threadgroup_barrier(mem_flags::mem_threadgroup);` |
| `Alloca{Shared}`（`threadgroup float x[N]`） | 函数体开头的 `threadgroup` 存储声明 + 指针绑定（与 thread 私有同构） |
| `Intrinsic::SimdReduce{op}` | `simd_sum(x)` / `simd_product(x)` / `simd_min(x)` / `simd_max(x)` |
| `Intrinsic::TileMma`（SimdGroupTiles 档） | 8×8×8 simdgroup_matrix 序列 |
| 派发形状 | threadgroup 宽度 = `policy.launch.threads[1]`（flat），网格不变 |
| `Intrinsic::ThreadId{dim:0}` | `gid.x`（local threadgroup 内序号，映射已把它定为 strip 基址） |

线程派发：Metal 的 threadgroup 尺寸来自 policy（编译期纯函数 `policy_for`），运行时按它派发；一个 workgroup 承担一个 tile，与 WGSL 的 workgroup_size 语义一致。

### 4.1 AIR 的两个实证约束（Metal 后端的硬边界）

Metal 的所有路径（含 WGPU：naga→MSL→AIR）最终都由 Apple 的 AIR 编译，三个已实证的约束（前两条是错编，`probe_metal.rs` 是复现与回归证据；第三条是性能）：

1. **64 位下标的读改写**：`thread` 数组上「`racc = t[i64 下标]; racc += …; t[同下标] = racc`」的 serial TileMma 累加被错误编译——累加塌缩进单个元素（8 变体二分定位；单变量 i64 下标与只读 i64 表达式下标均无恙）。**约束**：Metal emitter 的合成循环变量（serial mma 的 mo/no/ko、serial reduce 的 ro/ri）一律拼 `int`，与 WGSL 端的 i32 循环口径一致。**对第 2/3 期的警示**：simdgroup 归约与 MMA 的新拼写在第一版就要用 32 位循环变量，勿沿用 C 端的 int64 口径。
2. **fast-math 折叠常量无穷**：AIR 默认 `fastMathEnabled=on`，`x + (-INFINITY)` 被折叠为 `x`（L–N 变体；bitcast、`metal::select`、同块数组中转均存活失败，运算链侥幸存活不可依赖）。masked softmax 的 `qk + select(mask, 0, -inf)` 由此整个消失，输出与无掩码 kernel 完全一致。**双层防御**：主防御在 IR——instcombine 的 `recombine_masked_adds` 把该模式重写为 `select(mask, x, -inf)`（select 生产无穷而非把它喂给加法，fast-math 无法折叠运行时选择的值；代数恒等式精确）；WGSL 侧 belt 是 vendor patch 按特征串关闭 fast-math。belt 有一处已修正的事实：初版只匹配 `bitcast<f32>(0xff800000u)`，而 naga 在拼 MSL 前把该常量折叠成 `as_type<float>(4286578688u)`——旧串从未命中，belt 形同虚设；现在两串都匹配（`vendor/wgpu-hal/src/metal/device.rs` 两处 `CompileOptions` 位点）。gate 必须按需点火：safe-mode 的管线编译是驱动最慢路径，大 kernel 一次要分钟级，只有真携带无穷的 kernel 才付这个价。
3. **生成文本的确定性是编译性能的一部分**：Metal 驱动按「源文本 + 编译选项」为管线状态做持久缓存，冷编译在独立的 XPC 编译服务里进行——fmha 这类 kernel 一次 40–110 秒，safe-mode 下 85 秒到 6 分钟，而库编译（`new_library_with_source`）只 0.1 秒。`materialize_all`（`sile-lsir/src/lower/module.rs`）曾按 `recipes` 的 `HashMap` 迭代序物化 tile，值编号随进程漂移，拼写文本每进程不同，缓存**永不命中**——每个进程都重付全额冷编译，同一例子 5–7 分钟且方差极大。这条曾经被连环误判：怪过 i64 坐标数组、怪过 safe-mode AIR「编译爆炸」、怪过 wgpu 的 error-scope/poll——采样栈里的 `MTLCompilerScheduler::buildRequest → dispatch_sync → XPC wait` 只是缓存未命中冷编译的形态。修复是把物化序按 value 序号排序（注册序，即确定性的程序序）；验收即同一二进制连跑两次：冷 7:19 → 热 4.3 秒。**约束**：任何影响发射顺序或值编号的阶段，禁止依赖 `HashMap`/`HashSet` 迭代序（Rust 每进程随机种子）。

## 5. WGSL 拼写要点：64 位整数全端降级

WGSL 端的宽度口径与 C 端相反：IR 的 64 位整数在拼写层一律降到 32 位，生成代码里不出现 i64。证据都在代码注释与 `handwritten/_probe_*.wgsl` 探针里，记录如下。

**问题记录（三处证据）**：

1. **下标口径**：WGSL 下标只接受 `i32`/`u32`，lowering 按 LLVM 口径让每个下标是 `i64`。`wgsl_index`（`wgsl/codegen.rs`）在每个下标处拼 `i32(...)` 窄化，是两个口径唯一的交汇点；其余值保留 IR 宽度，i64 因此仍在生成代码里流动——tid 拼 `i64(local_index.x)`（strip 循环随之），memcpy 循环变量是 `i64`，select 的整常量带 `i64` 标注。
2. **能力拖累**：i64 在 WGSL 是可选能力。因为值里有 i64，适配器选择硬性要求 `SHADER_INT64`（`wgsl/runtime.rs`）——没有它的设备一个 kernel 都不能跑，报错措辞却像 kernel 有错；naga 验证被迫开 `Capabilities::all()` 放行 i64（`wgsl/mod.rs`），堵死了「生成代码意外回潮 i64 时本地编译被拒」的 fail-closed 通道。
3. **开销机制**：wgpu 路径的终点是 AIR，无 64 位整数指令。WGSL 的有符号溢出是回绕语义，naga 把每个 `i64 + i64` 拼成 `as_type<long>(as_type<ulong>(a) + as_type<ulong>(b))`——strip 循环下标算术的每一步加法变三条指令（`_probe_strip64.wgsl` 记录该拼写，`_probe_i64.wgsl` 在 Metal 上量化总开销）。

**决策**：一切 64 位整数在 WGSL 拼写层降到 32 位。下标与下标算术 `i32`（现状保留）；tid 与容量、位运算类 `u32`（`local_index`/`workgroup_id` 内建本就是 `u32`）；合成循环变量 `i32`（树归约的 ro/ri 与 serial mma 的 mo/no/ko 已是此口径，§4.1 第 1 条的 WGSL 侧对应）；形状与整标量描述符由 host 按 32 位装载，host 签名同步收窄。

**依据**：

- i64 在本管线的 WGSL kernel 里没有用户。一 tile 一 workgroup、workgroup 存储 16KB 上限（§2），可寻址元素数距 2^31 相去数量级。
- 降级后 `SHADER_INT64` 不再是运行前置，缺能力的设备回到可用集合；naga 能力集收回到默认，i64 回潮在本地编译期被拒，而不是在远端缺能力的设备上运行期被拒。
- 与 Metal 同一结论（§4.1 第 1 条）：AIR 无原生 64 位整数，32 位是两端共同的快路径。

**fail-closed 边界**：降级是语义改写，安全条件是值域契约「单 kernel 可寻址元素数 ≤ 2^31−1」。launch 形状在 host 手里，派发前检查、超出即拒绝派发，不在设备侧静默回绕；WGSL 的整数回绕虽有定义，契约内永不触及。

**兑现记录（全部落地）**：tid/块 id/select 常量/memcpy 循环的拼写收窄；描述符与 host 签名按 32 位装载（`i32::try_from` 值域契约，超限拒绝派发）；删去 `SHADER_INT64` 前置检查；naga 能力集收回默认（fail-closed：i64 回潮在本地编译期被拒）。验收：examples 全部 8 个 kernel 在 C/WGPU/Metal 三端数值对拍全绿，workspace 测试全过。

**兑现途中的一条 host 侧教训（descriptor 缓存撞 key）**：设备侧描述符 buffer 的缓存 key 原来只有字节长度。降级把整数标量描述符也压到 4 字节之后，float 标量通道与整数标量通道在长度上对齐——两个 4 字节描述符撞进同一个缓存项共享一块 buffer，后写的整数标量覆盖先写的 float 标量，`sv = 0` 使 `QK^T ≡ 0`、softmax 退化为均匀分布，输出精确等于 `mean(v)`。修复是把 label 并进缓存 key。教训：降级让「宽度即身份」的假设失效，凡按字节长度做等价判断的地方都要补上通道身份。

**遗留一例（未复现，记录在案）**：修复完成后的一次 `--all` 套件中，WGPU 端 causal 曾以 0.0733 的偏差失败一次（与「整数标量未生效、causal 按 plain 语义计算」的量级吻合），随后直跑 3 次、套件 6 轮均不复现。当时的怀疑链（描述符写与派发的队列序）推不出违规路径；窗口恰好跟在当日大量杀死的 XPC 编译进程之后，倾向驱动瞬态。再次出现时按独立回归处理：先 dump 双侧输出与描述符，再查 `write_buffer` 与 submit 的队列序。

## 6. 验证策略与已知弱点修复

### 6.1 每档的验证

- **数值**：examples 三端对拍（numpy oracle）；`SILE_SERIAL=1` 把并行映射与拼写分开归因。
- **形态**：`mapping_identity.rs` 每归约算子的单位元测试；新增 SimdReduce 形态测试。
- **性能**：`handwritten/` 手写参照定 80% 线（原则 5）；Metal 参照补 `.metal` 手写档。
- **契约**：parallel_contract 对每端跑；MSL 由 `newLibrary` 真编译闭环。

### 6.2 已知弱点的修复顺序（audit 三弱点）

1. **假 barrier**：`share_owner_tiles` 只在「翻入 shared 的 tile 存在真跨线程读者」时插 barrier；`parallel_contract` 从存在性配对升级为「每个 Shared alloca 的写→读之间有 barrier」的精确检查。在 Metal 路径数值全绿后再动，带回归测试。（实现 simd 两档时再次实证：fixture 里无跨线程读者的源 tile 也被翻 shared + 插 barrier。）
2. **静默降级可观测**：映射失败回退 serial 时输出 trace 警告（`SILE_TRACE` 门控）。已落地。
3. **自指契约（块名前缀探测 plumbing 循环）**：维持现状，记录为已知弱点——修复需要 lowering 与 mapping 共享一个显式的循环种类标注（新的跨阶段词），收益与风险不成比例，列入路线图。

### 6.3 实现兑现记录（simd 两档落地，2026-10）

两档已全部落地并通过验证闭环，与 §2 能力矩阵、§3.3/§3.4 的理论依据逐条对应：

- **四个新 intrinsic**（`lsir::Intrinsic`，映射植入、Metal 独有）：`SimdReduce{reduce}`（args [f32]，硬件 simd 归约并广播）、`SimdLaneId` / `SimdGroupId`（`thread_index_in_simdgroup` / `simdgroup_index_in_threadgroup`——simd 坐标必须用内建，tid→simd 映射是实现定义的，自行推导即错）、`SimdGroupMma{m,n,k,rhs_cols,row_groups,col_groups}`（args [lhs,rhs,acc?,out] 共享存储，result-less，分区参数自包含随指令走，不用 metadata）。
- **policy**：`ReducePolicy::SimdGroupTree`（W≤32 零 shared 零 barrier 直写；W>32 lane 0 写 scratch[simd_gid]，W/32 个 partial 走小树）；`MmaPolicy::SimdGroupTiles`（Goto & Geijn 寄存器分块，每 simd-group 一个寄存器块；carry staging → barrier → 幸存 intrinsic → barrier → strip 拷回 → 尾 barrier）。
- **可行性单一实现**：`simd_tiles_feasible(func, threads, shared_cap)` 供 backend policy 与 reduce/mma stage 共用，两端判定不一致在结构上不可能。
- **拼写**：metal/codegen 按 §4 拼写表展开；simd 内建出现在 Metal 之外即契约拒绝（fail-closed 三层：输入契约拒绝 lowering 产的 simd 形态、`parallel_contract` 按 target 放行、naga/cc/`newLibrary` 端内验证）。
- **验证**：lsir 5 个形态测试（simd_mapping.rs：树折叠、W≤32 免 scratch、中继小树、mma 分区 2×2、可行性镜像）+ backend 1 个契约测试（含 C 的全目标拒绝）；examples 三端 8×8 数值对拍全绿（SimdGroupTree 在 AMD 上跑通 softmax/rms_norm/flash_attention）；gemm 在本机正确降级 ThreadStrips 且数值正确（Apple7 探测生效）——降级阶梯的完整实证。
- **未决**：SimdGroupTiles 的数值正确性需 Apple7+ 机器验证（本机只能验证降级路径）；软件流水与 §7 路线图项不受影响。

## 7. 路线图（不在本次范围）

| 项 | 触发条件 | 理论锚点 |
|---|---|---|
| 软件流水（gemm k 循环双缓冲） | 三端全达标后仍要逼近手写 gemm 上限 | modulo scheduling（Rau, *Iterative Modulo Scheduling*, 1994）；Triton pipeliner |
| WGSL subgroup | wgpu/naga subgroup 特性稳定 | 与 SimdGroupTree 同构， WGSL spec subgroup 扩展 |
| per-value layout 系统 | 出现 strip 模型表达不了、达标线以下的 kernel | Triton blocked layout 三元组 + convert_layout |
| PTX M4（mma.sync + shared 布局） | BACKEND_PLAN 既定里程碑 | PTX ISA；fragment 布局与 simdgroup_matrix 同族 |
| Metal float4 合并访存 | saxpy 类 kernel 带宽不足时 | 合并访存规范（相邻线程相邻地址） |
