# Tile IR → LSIR → backend 计划书

状态：执行中（2026-10-22 刷新）。M0、M1、M2 已完成；M2.5（标量提升）、M3（映射 pass 族 + WGSL）已完成；M4 未开始。pass 拓扑的规范裁决见 `docs/PIPELINE_SPEC.md`。前身结论见 `crates/sile-tile-ir/TILEIR_ALIGNMENT.md`（Tile IR 文本已与 cutile-rs 逐字节对齐，17/17）。

## 1. 目标

把编译管线从 arena Tile IR 接到可执行后端，为 `examples/` 里的每个计算生成与手写最优实现同级性能、语义一致的代码：

```
TypedKernel → arena Tile IR（冻结）→ LSIR（重建）→ C / CUDA(PTX) / WGSL（改造）
```

量化验收：

1. `examples/` 的 8 个性能目标（saxpy、softmax、rms_norm、gemm、gemm_static、batch_matmul、flash_attention、flash_attention_causal）在 C 与 WGSL 两个 target 上，与手写参照对拍数值一致、性能达到手写版的 80% 以上。
2. Tile IR 导出保持 17/17 逐字节对齐，`cutile::emit` 不动。
3. TileIrFunction（function_ir）与 `lower_to_tile_ir` 删除，编译链上只剩一个 Tile IR。

## 2. 原则

1. **arena Tile IR 是唯一标准输入，冻结。** 它是 cutile-rs 的 `cutile-ir` 的移植，文本导出逐字节对齐。下游不许绕过它，也不许为迁就下游而改它。
2. **下游全部可大改，不写兼容桥。** TileIrFunction 直接退役，不做 desugar 过渡层。旧 lowering 不逐行翻译，算法可搬的搬，骨架重建。
3. **tile 级语义保留到最后一刻。** reduce、mma 是 tile 上的集合操作，不在 lowering 时拆散成标量循环。LSIR 保留它们为 tile 级原语，由 backend 按 target 选并行形态。这条是并行归约的前提：归约的 tile 级语义一旦在 lowering 摊平，backend 就再也无力并行它（现状 `fused_reduce` 零触发的根因）。
4. **映射决策集中在一层。** tile→线程/循环/共享内存的决策只在 device lowering 做；backend 是方言拼写器（C/WGSL/PTX 语法、验证、运行时）。cutile 的 `tileiras` 承担的角色由这一层承担。
5. **手写参照定标。** 每个性能目标先写尽力最优的 C/WGSL 参照，作为数值 oracle、形态靶子和性能基准。没有参照就没有"类似性能"的定义。
6. **最好的代码是没有代码。** 复用已有的：LSIR pass 家族（cse/licm/instcombine）、backend 的结构化 CFG 发射器与四方言、naga 验证、四个运行时、bench.py/compare.py。

## 3. 现状与差距

### 3.1 参考源

cutile-rs 在 Tile IR 之后没有任何可执行自研路径：它把 Tile IR 字节码交给闭源 `tileiras` 换 cubin，拒绝生成 PTX，没有解释器、LLVM、WGSL。tile→线程映射、shared memory、barrier 全在黑盒里。可搬的只有：host 侧 grid 推导（`ceil(extent/tile)`，已对齐）、launch check 宿主求值（已搬为 `predicate::LaunchCheck`）、两级 JIT 缓存键设计、fixtures 的 IR 形态作语义样本。**device 端全部自研。**

### 3.2 管线状态（2026-10-21 刷新）

现链路：`TypedKernel → cutile::emit → arena ir::Module → lower_module_to_lsir → LSIR passes（13 项）→ sile-backend`（编排见 `crates/sile/src/compiler.rs`）。

- **断层一（双轨）——已解决**：`TileIrFunction` 与 `lower_to_tile_ir` 已删除（a63ecb3），编译链上只剩 arena Tile IR；cutile-rs 的判定上移为 sile-hir passes（d101b31），emitter 只消费判定。
- **断层二（质量）——仍存在**：WGSL `workgroup_size(1)`、PTX 一 block 一线程；shared memory 全程未用；barrier 仍被 `verify_gpu_contract` 无条件拒绝；`VecLoad`/`VecStore` 已定义但 lowering 从不发射；TileMma 在 C/WGSL/Metal 为串行循环、在 PTX 未实现。映射决策的收拢方案见 `docs/PIPELINE_SPEC.md`。

### 3.3 一个重要资产

arena Tile IR 的 for/if/reduce 是真 region。新 lowering 是结构化树遍历，不需要旧 `lower/block.rs`（1072 行）那套"线性 inst 流 → 结构化 CFG"的逆向恢复。直接退役反而更省。

## 4. 方案

### 4.1 目标管线

```
TypedKernel（HIR，不动）
   │ cutile::emit（不动，17/17 对齐）
   ▼
ir::Module                      arena Tile IR，唯一、冻结
   │ device lowering（重建：结构化遍历 region 树）
   ▼
sile_lsir::Function          物理化设备 IR（形态扩展，见 4.2）
   │ 既有 LSIR passes（cse / licm / instcombine / sroa / ...，保留）
   ▼
sile-backend（改造物化规则，见 4.3）
   ├─ C    → cc -O2 -fopenmp → .so → libloading（运行时已有）
   ├─ PTX  → NVIDIA 驱动（cuda）
   ├─ WGSL → naga 验证 → wgpu（验证已有）
   └─ Metal（既有）
```

### 4.2 LSIR 形态：物理化设备 IR

sile_lsir::Function 保留为 lowering 的目标语言（backend 的输入契约不变，方言层少动），扩展其原语面：

| 原语 | 现状 | 改动 |
|---|---|---|
| tile 数据 | `alloca [R][C]` + `ParamAbi::tile` | 保留；新增存储类别决策（私有 / Shared） |
| `Intrinsic::BlockId`/`ThreadId`/`Barrier` | 枚举已有，从不发射 | 启用发射 |
| `AddressSpace::Shared` | 枚举已有，从不发射 | 启用发射（WGSL `var<workgroup>`、PTX `.shared`、C 无此需求） |
| 归约 | `VecLoad`/`VecReduceAdd`（窄义、零触发） | 新增 `TileReduce { op: Sum/Max, dim, identity }`：tile 级归约原语 |
| 矩阵乘 | 无（lowering 时拆散） | 新增 `Mma { m, n, k }`：tile 级乘累加原语 |
| 访存 | 标量 gep | 加 `VecLoad`/`VecStore` 的真实发射（连续且宽度匹配时） |
| token | 无对应 | CPU：消除（顺序执行）；GPU：跨线程依赖处翻译为 Barrier |

原则重申：elementwise 摊到标量/循环；集合操作（TileReduce/Mma）保持 tile 级，把"怎么并行"留给 backend。 lowering 负责的回答只有三个：每个 tile 值存在哪、哪个维度分给线程、哪里需要 barrier。

### 4.3 backend 物化规则（并行归约的落点）

| 原语 | C（cpu） | WGSL | PTX（cuda） |
|---|---|---|---|
| tile 存储 | 栈数组，SROA 消中转 | `var<workgroup>`（协作值）或函数作用域（线程私有值） | `.shared` / 寄存器 |
| `TileReduce` | `#pragma omp simd reduction(...)`，交给编译器向量化 | workgroup 树归约：shared 数组两两折叠 + `workgroupBarrier()`；小行宽退化为每线程一行 | warp shuffle 归约，跨 warp 走 shared + `bar.sync` |
| `Mma` | 寄存器分块 micro-kernel（如 4×8 累加），auto-vec | 每线程子块 FMA 展开，shared 分块加载 | `mma.sync.m16n8k8`（Tensor Core） |
| tile→线程 | 一 tile = 一次 OMP 迭代，tile 内循环展开+SIMD | 一 tile = 一个 workgroup，`workgroup_size(N>1)`，元素摊给线程 | 一 tile = 一 block，warp 级映射 |
| grid | `ceil(extent/tile)`（host 已有） | 同 | 同 |

### 4.4 退役清单（直接删，无桥）

删除：

- `sile-tile-ir/src/lower.rs`（`lower_to_tile_ir`，2898 行）
- `sile-tile-ir/src/function_ir.rs`（1407 行）
- `sile-tile-ir/src/print.rs`（function_ir 的文本打印）
- `sile-tile-ir/src/passes/`（dce / pointwise_fusion / tile_expr_cse / verify——均吃 TileIrFunction）
- `sile-lsir/src/lower/*`（旧 lowering，~7800 行）

改造：

- `sile-lsir/src/lower/`：重建为吃 `ir::Module` 的新 lowering（入口 `lower_module_to_lsir(&ir::Module) -> Function`）。
- `crates/sile/src/compiler.rs`：管线编排去掉 HirToTileIr/TileIrToLsir 的 function_ir 段，返回 `(ir::Module, Function)`。
- `crates/sile/src/dump.rs`：Tile IR 阶段 dump 改为 `to_mlir_text()`；TileIrPassKind 段移除。
- `crates/sile/src/kernel_launcher.rs`：`CompiledParts` 的 TileIrFunction → 诊断文本或 `ir::Module`。
- `crates/sile/src/lib.rs`：`tileir` 模块 re-export 清理。
- `sile-backend/src/verify.rs`：解除对 Barrier 的无条件拒绝，改为"有 Shared/Barrier 时校验配套约束"。

保留不动：HIR 全家、`cutile::emit`、arena `ir`、LSIR passes、backend 方言发射层、四个运行时、scripts 工具链。

### 4.5 阶段与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0（并行推进） | 手写 golden：saxpy / softmax / rms_norm / gemm 的 C 与 WGSL（参数化尺寸：小尺寸自检、大尺寸 bench），对拍骨架 | 参照自身数值正确、可 bench |
| M1 | 退役 + 新 lowering：结构化遍历，elementwise→reduce→mma 顺序实现，C 端先全通；启动 4.4 全部删除 | examples 在 C target 数值对拍全绿；17/17 对齐不动 |
| M2 | C 端质量：TileReduce→`omp simd reduction`、tile 中转拷贝消去、VecLoad 触发 | saxpy/softmax/rms_norm 达手写 C 的 80%+ — **已完成 2026-10-21：99.5% / 87.1% / 86.7%** |
| M2.5 | 标量提升 pass（`ScalarPromote`）：私有标量槽 → 块参数 | PTX 不再每迭代 `ld.local`/`st.local`；四端数值不变 — **已完成 2026-10-21** |
| M3 | 映射 pass 族（`crates/sile-lsir/src/mapping.rs`，dump 段 `lsir-15..19-mapping-*`）+ WGSL workgroup_size>1、`var<workgroup>`、树归约 + barrier | softmax/rms_norm/gemm 达手写 WGSL 的 80%+，naga 验证过 — **已完成 2026-10-22：同窗口 A/B softmax 96–103% / rms_norm 112–128% / gemm ≈100%**。**2026-10-23 重测（compare_golden 5 轮取最优，新参照线）**：C 端 saxpy/softmax/rms_norm/gemm = 95.8/153.5/98.1/96.9% 全 OK；WGPU 端 = 50.1/53.8/108.1/25.7%，除 rms_norm 外 SHORT。sile 侧对父提交全部变快（softmax 2.0x / rms_norm 1.6x / gemm 1.4x，i32 降级的收益），差距来自 bench 刷新后参照线抬高与 workgroup 粒度（如 softmax 每 tile 16 元素仅 8 线程 vs 手写每行 256），WGPU 差距为后续工作基线 |
| M4 | PTX/cuda：shared 布局、`mma.sync`、barrier；host 侧搬 cutile-rs JIT 缓存键 | gemm 对齐手写 CUDA 参照 |

顺序理由：C 先行（运行时最简单、迭代最快）；WGSL 次之（naga 编译期验证兜底，"多线程 tile"的语义模型在它上面踩对）；PTX 最后（方言最重，模型对了只剩拼写）。

### 4.6 风险与取舍

1. **退役期间 WGSL/PTX 退化**：M1 完成时 WGSL/PTX 输出暂为串行正确版（tile 物化规则只做了 C 端），M3/M4 恢复。期间 bench 的 GPU 对比列不可用。
2. **Tile IR 级 pass 暂失**：dce/fusion/cse 曾在 TileIrFunction 上跑。M1 后由 LSIR 层既有 pass 兜底；arena IR 上的 pass（含 cutile 的 dominance/verify）在后续版本重建，不阻塞主线。
3. **验收线分 kernel 设定**：gemm 的手写上限（blocked+micro-kernel / Tensor Core）远高于朴素实现，80% 线以各 kernel 的手写参照为准，不做全局一刀切。
4. **Mma 原语的抽象层级**：`Mma{m,n,k}` 若在 C/WGSL 上退化实现成本过高，允许 lowering 对非 PTX target 提前拆为 FMA 循环，但拆解必须发生在 backend 可见的 tile 级 op 之后（保留未来换形态的自由），此例外需在 lowering 中以显式开关呈现，不得静默。

## 5. 立即执行项（2026-10-22 状态：1–6 已完成；下一步为 M4）

1. M0 前四个 kernel 的手写 C/WGSL + 对拍骨架（`handwritten/`）。
2. M1 新 lowering 骨架 + saxpy 端到端（arena Module → LSIR → C → 数值对）。
3. 4.4 退役清单执行。
4. softmax / rms_norm（TileReduce 并行归约）→ gemm（Mma）依次接入。
5. 标量提升 pass（M2.5）已落地：`sile-lsir/src/passes/scalar_promote.rs`，链序 `Sroa → ScalarPromote → Dce`。
6. M3 已落地：映射 pass 族在 `crates/sile-lsir/src/mapping.rs`（五阶段 `policy/threads/reduce/mma/cleanup`，`MappingPolicy` 参数化），`launch.rs` 已接线；WGSL 侧解除 `verify_gpu_contract` 对 Barrier 的无条件拒绝，改为有 Shared/Barrier 时校验配套。阶段间的两个词 `Metadata::OwnerOrdered` / `Metadata::ReduceRelay` 由 builder 显式打戳，模块头写明了它们的存活规则。
7. Metal simd 两档已落地（2026-10-23，详见 MAPPING_DESIGN.md §2/§3.3/§3.4/§6.3）：`ReducePolicy::SimdGroupTree` 与 `MmaPolicy::SimdGroupTiles`，四个 Metal 独有 intrinsic（`SimdReduce`/`SimdLaneId`/`SimdGroupId`/`SimdGroupMma`），`simdgroup_matrix_supported()` 以 Apple7 探测进 policy——实证 simdgroup_matrix 在非 Apple7 设备（AMD Radeon Pro 5500M）上前端编译通过但 pipeline 编译失败，capability 必须在映射前决策。本机 AMD 上 SimdGroupTree 数值验证通过，SimdGroupTiles 正确降级 ThreadStrips（gemm 数值正确）；Apple7+ 上的数值验证待机器。
8. 下一步 M4：PTX/cuda（shared 布局、`mma.sync`、barrier）。
