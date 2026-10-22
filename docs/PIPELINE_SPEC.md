# 编译管线规范（两条路线）

本文档是 2026-10-21 grilling 定稿的规范与决策记录。事实基础当天核验：`scripts/check_tileir_alignment.sh` 实跑 17/17 PASS；两侧 fixture 目录逐字节一致。

- 路线 1：前端 → 宏解析 → HIR → TileIR。对齐契约，唯一硬标准是 diff=0。
- 路线 2：TileIR → LSIR → 端码（C/WGSL/PTX/Metal）。pass 拓扑，参照系 Triton 三层式。

里程碑与性能验收仍以 `docs/BACKEND_PLAN.md` 为准；本文档只管结构与不变量，两者冲突时以本文档为准并回改计划书。

## 1. 路线 1：对齐契约

### 1.1 核心不变量

**TileIR 冻结；sile 导出的 TileIR 文本与 cutile-rs 同源程序逐字节相等（diff=0）。这是路线 1 的唯一硬标准。**

由此推出的裁决规则：

1. 任何导致文本不一致的差异都是回归，修 sile 使 diff=0 恢复。没有「有意分叉但文本不同」这个类别。
2. 不改变文本的层级安排不算分叉。判定上移 sile-hir（d101b31）即属此类：emitter 换了判定来源，文本不变。
3. cutile-rs 是外部参照项目，不是本工作区所有物，**不修改它**。`crates/sile/tests/fixtures/tileir/` 是对齐快照的入库真相源；上游演进需要重新同步时，显式重导 fixtures 并在提交信息注明来源 revision。

### 1.2 分工（对应 cutile-compiler 六段 AST pipeline）

| cutile-rs 规范段（`cutile-compiler/src/passes/mod.rs`） | sile 对应物 |
|---|---|
| Name Resolution（per module） | rustc 本身；泛型由 rustc 单态化 |
| Proof Analysis | `sile-hir/src/passes/{access,bounds,predicate}.rs` 五级证明阶梯 |
| Node IDs | `sile-hir/src/passes/value_ids.rs` |
| Type Inference（DSL 窄推断） | `sile-hir/src/typeck.rs` |
| Typed Dispatch Lowering | `sile-macros/src/frontend/lower.rs` + `inline.rs`（设备调用内联） |
| IR Emission（只消费判定） | `sile-tile-ir/src/cutile/emit.rs` |

emitter 只消费判定、不做判定（cutile `proof_analysis.rs` 的铁律，`TILEIR_ALIGNMENT.md:106-121` 有对应记录）。

### 1.3 门禁（双层）

1. **全量**：`scripts/check_tileir_alignment.sh` —— 17 个 example 全导出，逐 fixture 与入库快照 `left == right` 判字节相等。
2. **快速**：`cargo test` 内的最小 kernel 对拍测试（`crates/sile/tests/tileir_fixture.rs`）——用内部 API 程序化构造最小 kernel，走 `emit` + `to_mlir_text` 与快照比对。让日常 `cargo test` 也能抓大漂移，不依赖手动跑脚本。

两层比对的都是入库快照。快照更新的唯一合法途径：diff 确认变化来自预期改动，重新导出并提交。

## 2. 路线 2：pass 拓扑

### 2.1 参照系：Triton 三层式

| Triton | sile 对应物 | 状态 |
|---|---|---|
| TTIR（结构化 tile 级，语义冻结） | arena TileIR，零 pass，1 个 dominance 校验 | 已就位，冻结 |
| TTGIR（布局分配、线程映射、shared、convert_layout） | `crates/sile-lsir/src/mapping.rs` 的映射 pass 族：`MappingPolicy` 参数化五阶段 `policy/threads/reduce/mma/cleanup`，dump 段 `lsir-15..19-mapping-*` | 骨架就位（M3，2026-10-22；WGSL 单端验证子集，措辞边界见 §2.1） |
| 标量 LSIR + 优化（对标 Triton 的 LLVM IR 槽位） | 自研 SSA `sile_lsir::Function` + 14 pass | 已就位，链序原则见 §2.2 |
| 各端 codegen | `TargetDialect` 拼写器 + per-target 契约 verify | 已就位，保持 |

**LSIR（Lowered Scalar IR）**：路线 2 中段 IR 的本名（2026-10-21 更名，原名「LLVM IR」是误称）。它是自研 SSA IR（`sile_lsir::Function`）：无 libllvm 依赖、不产 bitcode、喂不进 NVPTX/llc。文本形式模仿 LLVM IR（print 的目标是 `llvm-as` 可接受，intrinsic 拼写沿用 `@llvm.*`）——这是格式选择，不是依赖；保留文本兼容是未来接真 LLVM 后端的钩子。在那之前，标量优化全部由 LSIR 上的 pass 自担（这也是「M4 前需自写标量提升」的根因）。

TileLang 的启示只取一条：TileReduce/TileMma 这类 tile 级原语必须活到映射层（BACKEND_PLAN 原则 3），不引入 schedule 概念。措辞边界（审计 2026-10-22）：TileLang 的原语与 schedule 是一体两面，本管线取的是「tile 原语独立于线程映射存在」这半面；映射族的 shared/staging/树归约手段与 TileLang 的 schedule 思想同源，不声称只受 Triton 影响。

映射族与 TTGIR 的对应关系（措辞降级，审计 2026-10-22）：映射族是 **WGSL 单端验证过的 TTGIR 子集**——有线程映射、shared staging、树归约、barrier 契约；没有 per-value layout、convert_layout、swizzle、布局推断（policy 是每 target 全局常量）、pipelining/warp specialization。说「TTGIR 对应物已就位」时读作：骨架就位，布局系统未建。

### 2.2 拓扑定稿

```
TypedKernel
  │ HIR passes：VerifyInput → TypeCheck → VerifyOutput
  ▼
arena TileIR（冻结；0 个变换 pass，仅 verify_dominance）
  │ lower_module_to_lsir：唯一下降器，零 phi（alloca+load/store 表达循环）
  ▼
sile_lsir::Function
  │ 映射/物化 pass 族（mapping.rs，参数 = target policy）
  │   阶段：policy → threads → reduce → mma → cleanup（各一段 dump）
  │   决定：workgroup/block 尺寸、tile 维→线程、
  │         TileReduce 树归约+barrier、TileMma→mma.sync、Shared 布局
  │   不变量：映射只改放置，不改语义（抄 cutile：hoist never changes
  │         whether a kernel traps）。dump 段命名 lsir-1x-mapping-*
  │ 现有 14 pass，链序及排序原则：
  │   VerifyInput → Canonicalize → LoopSimplify → PhiAlias → Cse
  │   → InstCombine → LoopUnroll → InstCombine → Licm → Sroa
  │   → ScalarPromote → Dce → Cse → VerifyOutput
  │
  │   链序原则（审计定稿 2026-10-22，依据 docs/ACADEMIC_GROUNDING_AUDIT.md）：
  │   1. 验证边界：结构合法性入口验一次、终态验一次（VerifyInput/VerifyOutput），
  │      中间 pass 只依赖各自的局部前提。
  │   2. 先规范化、再分析：Cse/InstCombine 在循环分析之前，让循环体的
  │      指令数与形式稳定（LoopSimplify/LoopUnroll 都吃规范化后的形状）。
  │   3. 循环先成形、再动内容：LoopSimplify（preheader/单回边）先于一切
  │      需要插入 preheader 的 pass（LoopUnroll/Licm）。
  │   4. 标量提升最后做、一次做够：LoopUnroll 展开后槽的赋值点全部可见，
  │      Sroa（数组槽）→ ScalarPromote（标量槽）按可提升面从大到小收尾；
  │      提前提升会在 unroll 前引入不必要的块参数连锁。
  │   5. 删除性 pass 靠后：Dce/Cse 在所有引入新指令的 pass 之后，只关尾灯。
  │   6. barrier 安全：映射族在标量管线之后由后端调度（backend compile_inner），
  │      barrier 不暴露给任何可 hoist/可删除的标量 pass；标量端安全网是
  │      canonicalize 的不可移除表与 cse 的写-kill 全清单（含 Barrier）。
  ▼
各端：TargetDialect 拼写器 + 契约 verify（fail-closed）
  ├─ C    cc -O2 -fopenmp   ├─ WGSL naga→wgpu
  ├─ PTX  自写发射器         └─ Metal
```

### 2.3 已裁决的三项

1. **映射决策收拢为独立 pass 族**（M3，2026-10-22 完成）。BACKEND_PLAN 原则 4「映射决策集中在一层」由它兑现：`MappingPolicy`（`launch`/`reduce`/`mma`/`vector`/`shared_cap`）是数据，五阶段各自只读它。vector hint（`emit.rs:2256-2283`）等映射类判定随 M3 收拢进 policy。
   跨阶段通信只靠两个显式词：`Metadata::OwnerOrdered`（槽是读者自己的，builder 已验 `extent % threads`）与 `Metadata::ReduceRelay`（槽是 `TileReduce` 的广播中继）。二者由 builder（reduce 阶段，或 lowering 发射该访问处）打戳，下游不再从指针链反推。存活规则写在 `mapping.rs` 模块头：保留指令的 pass（`loop_unroll` 克隆、`canonicalize` 拷贝）保留元数据，`cse` 拒绝合并带元数据的指令，新造指令的 pass 自己决定（`storage` 的中继转发沿用被换下的 load 的戳）。
2. **零 phi 契约的 PTX 结局 = IR 层标量提升**（2026-10-21 落地，G2）。lowering 保持零 phi 不变；`passes/scalar_promote.rs` 在 Sroa 之后、Dce 之前，把私有标量槽提升为块参数（phi 的块参数形）。放置用迭代支配边界，按活跃性剪枝，再沿支配树重命名。PTX 发射器不再每迭代 `ld.local`/`st.local`：saxpy 的 `.local` 只剩一个 tile 数组，四端数值对拍不变。
   发射侧配套：`emit.rs` 的计数循环识别同时认 alloca 形（`iv_is_param = false`）与提升形（`iv_is_param = true`），两种形态都拼 `for` 并保留 `#pragma omp simd`/向量化提示；后端契约 `verify_structured_shapes` 因此不再误拒提升后的循环。
3. **emit 层保持拼写器 + 契约 verify**。per-target verify 拒绝方言拼不出的形态，方向正确，保持 fail-closed。

### 2.4 字面回答「经过多少 pass」

- TileIR → 标量 IR：1 个下降器，TileIR 上 0 个 pass（冻结期不变）。
- 标量 IR → 端码：现有 14 pass + M3 映射族；每端 1 个拼写器 + 1 个契约 verify。

## 3. 决策记录（2026-10-21）

| # | 裁决 | 内容 |
|---|---|---|
| 1 | 路线 2 参照系 | Triton 三层式；缺的映射层即 TTGIR 对应物 |
| 2 | 零 phi 的 PTX 结局 | IR 层标量提升，非发射器内提升 |
| 3 | 映射层定位 | 独立 pass 族 + 独立 dump 段，非藏于 lowering/emit |
| 4 | 路线 1 门禁 | 最小 Rust 对拍测试 + 脚本全量门禁，双层 |
| 5 | cuda-rust | 外部参照，不修改；sile 快照为入库真相源 |
| 6 | 对齐哲学 | diff=0 是核心；文本不一致即回归，无例外 |
| 7 | 文档修缮 | 四处失真即日修复（脚本注释、BACKEND_PLAN、lower 注释、dumps 残留） |
| 8 | 产出 | 本规范落盘 docs/，作后续实现与 review 的锚 |
