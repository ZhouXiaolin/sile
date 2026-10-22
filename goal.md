# goal：路线 2 执行（LSIR → 端码）—— 2026-10-21 grilling 定稿

权威文档：`docs/PIPELINE_SPEC.md`（拓扑与不变量，冲突时以它为准）· `docs/BACKEND_PLAN.md`（里程碑与性能验收）· `docs/LOWER_DESIGN.md`（下降契约）。

## 0. 已完成的前置（2026-10-21）

- **LSIR 更名**：`sile-llvm-ir` → `sile-lsir`（crate 目录、`LsirFunction`/`LsirPassKind`、`lower_module_to_lsir`、dump 段 `lsir-XX`、`SILE_PRINT_LSIR`）。文本形式模仿 LLVM IR（print 目标为 `llvm-as` 可接受、intrinsic 拼写 `@llvm.*`）是格式选择，保留为未来接真 LLVM 后端的钩子。更名后 cargo test 全绿 + 17/17 对齐。
- **路线 1 门禁双层**：`scripts/check_tileir_alignment.sh`（17 fixture 全量）+ `crates/sile/tests/tileir_fixture.rs`（cargo test 抓漂移）。
- **规范与文档**：PIPELINE_SPEC.md 落盘；BACKEND_PLAN 状态刷新；四处文档失真修复；dumps/ 旧管线残留（785 个 tileir-01..05）清理。

## 1. 已定裁决（全文见 PIPELINE_SPEC §3）

1. 参照系 Triton 三层式；缺的映射层即 TTGIR 对应物。
2. 零 phi 的 PTX 结局 = IR 层标量提升（非发射器内）。
3. 映射决策收拢为独立 pass 族 + 独立 dump 段。
4. 门禁双层（脚本 + Rust 测试）。
5. cuda-rust 是外部项目，禁止修改；sile fixtures 为入库真相源。
6. diff=0 是唯一硬标准；文本不一致即回归。
7. 文档失真即日修。
8. 规范落盘 docs/，本 goal 是执行面。

## 2. 目标

把路线 2 从「C 端正确」推进到「三端各达手写参照 80%」，全程不破坏 diff=0。

## 3. 任务分解（顺序即依赖）

### G1 · M2 收尾：C 端质量 ✅（2026-10-21 完成）
- `VecLoad`/`VecStore` 真实发射（lowering 在连续且宽度匹配时发射；三端拼写已备）。
- 验收：saxpy/softmax/rms_norm ≥ 手写 C 的 80%（`compare_golden.py`）；17/17 对齐不动。
- **结果**：26 测试全绿、17/17 PASS、`compare_golden.py --rounds 5` 实测 saxpy 99.5% / softmax 87.1% / rms_norm 86.7%（均 ≥80%）。注：早前 100.5%/99.9%/193.3% 与本轮数字差异来自机器负载与两侧同为 median-of-100 的同频条件，非编译产物差异。

### G2 · 标量提升 pass（排在 M4 前，可先于 WGSL）✅（2026-10-21 完成）
- LSIR 新 pass：Sroa 之后、Dce 之前；引入 phi 或块参数；复用 `passes/cfg.rs` dominance 基础设施。
- 验收：PTX 端循环归纳变量不再每迭代 `ld.local`/`st.local`；四端数值对拍不变。
- **实现**：`crates/sile-lsir/src/passes/scalar_promote.rs`，块参数形 phi。放置 = 迭代支配边界 + 活跃性剪枝；重命名沿支配树。dump 段 `lsir-11-scalar-promote`。
- **发射侧配套**：`emit.rs` 的 `AllocaForLoop` 泛化为 `CountedForLoop`（`iv_is_param` 区分两形）；`for` 拼写与 `#pragma omp simd` 在两形下都保留；`verify_structured_shapes` 因而不误拒提升后的循环。
- **结果**：九个代表性 kernel（saxpy/softmax/rms_norm/gemm/dropout/flash_attention/flash_attention_causal/hoist/static_extent）的标量 alloca 从 6..98 个降到 **0**；saxpy 的 PTX `.local` 只剩一个 tile 数组，循环内归纳变量在寄存器间传递；17 个 example C 端全绿、17/17 对齐、`cargo test --release` 29 项全绿。
- **过程中的两处修正**（都是 pass 的正确性缺陷，非发射器问题）：must-analysis 的数据流初值必须从 top（已赋值）向下收敛，否则循环回边让它永远停在 false；活跃性剪枝必须做，否则循环体内定义的槽会把参数插到外层循环头。

### G3 · 映射 pass 族 + WGSL（BACKEND_PLAN M3）
- 映射 pass：policy 参数化（workgroup/block 尺寸、tile 维→线程、TileReduce 树归约+barrier、TileMma 展开、Shared 布局）；dump 段 `lsir-1x-mapping-*`；不变量「只改放置，不改语义」。vector hint（`emit.rs`）随此收拢进 policy。
- WGSL：`workgroup_size>1`、`var<workgroup>`、树归约 + `workgroupBarrier()`、token→barrier；解除 `verify_gpu_contract` 对 Barrier 的无条件拒绝，改为有 Shared/Barrier 时校验配套。
- 接线 `launch.rs`（`sile-lsir/src/lib.rs` 声明 `pub mod launch`）。
- 验收：softmax/rms_norm/gemm ≥ 手写 WGSL 的 80%；naga 验证过；17/17 不动。

### G4 · PTX/cuda（BACKEND_PLAN M4）
- shared 布局、`mma.sync`（依赖映射层的线程 fragment）、barrier；TileReduce/TileMma 在 PTX 物化。
- host 侧搬 cutile-rs JIT 两级缓存键设计。
- 验收：gemm 对齐手写 CUDA 参照。

## 4. 不变量（每个任务验收时逐条对照）

1. **diff=0**：任务完成后 `check_tileir_alignment.sh` 17/17 与 `tileir_fixture` 测试必须绿。
2. TileIR 零变换 pass；lowering 保持零 phi（标量提升是 lowering 之后的 pass，不违契约）。
3. 映射只改放置不改语义；backend fail-closed。
4. cuda-rust 禁改；fixtures 重导必须注明来源 revision。
5. 数值回归用 `scripts/compare.py --before/--after` 与 `numpy_ref`；性能一律 median-of-100（本机冷启动瞬态与 best-of 时钟谎言已定性，勿回退）。
