# TileIR → Backend Pass 管线的学术支撑裁决

> 2026-10 审计。方法：三路独立审计（设计层、映射族、标量管线逐行），主上下文对每条重指控回源核验。
> 裁决用词：**grounded**（有文献/成熟系统对应且实现不失真）、**principled**（工程理由自洽但无文献对应）、**fitting**（为通过当前 fixture/校验器而定制的假设，学术上不成立）。

## 总裁决

**不是根据答案凑出来的，但也不是处处有学术支撑。**

- 标量 pass 链的核心算法（SSA 构造、支配边界 phi 放置、pruned-SSA、must 数据流、自然循环）逐行核对后与 Cytron et al. TOPLAS 1991 及标准教材算法一致，且多处带有超出「凑 fixture」需要的防御性检查（操作数支配性预检、unreachable 放弃、hops 上限防环）。**grounded**。
- 映射族的内存序硬部分（写→barrier→读、barrier 置于均匀汇合块、树归约每步 barrier）正确且与 CUDA/WGSL 内存模型一致。**grounded**。
- 但存在三类系统性弱点，均属于 **fitting**：
  1. **自指契约**：映射 pass 靠硬编码块名前缀（`is_plumbing_header`，sile-backend/src/mapping.rs:1449-1457）和窄形状匹配（`find_strip_loop`）识别自家 lowering 产物；
  2. **校验器与 pass 互相配合**：`share_owner_tiles`（sile-lsir/src/mapping.rs:1195-1220）在无跨线程访问处也插 barrier 以喂 `parallel_contract` 的 has_shared==has_barrier 存在性检查——校验是必要非充分的装饰；
  3. **信任自盖章元数据**：带 `OwnerOrdered` 戳的 gep 跳过 affine 验证（mapping.rs:1130-1133），戳本身由映射族自己打上。

## 分项裁决

### 标量 pass 链（14 pass，sile-lsir/src/passes/）

| Pass | 裁决 | 依据与缺口 |
|---|---|---|
| scalar_promote | **grounded** | dominance_frontiers（scalar_promote.rs:399-432）是标准 Cytron runner 算法；placement（:437-463）标准迭代支配边界 worklist；prune_to_live_in（:489-）标准 pruned-SSA；rename_block（:565-）支配树栈式重命名。assigned_on_entry must 分析（:247-292）sound。缺口：测试仅 3 个，无 diamond-CFG/不可归约 CFG/部分初始化路径覆盖 |
| sroa | **grounded** | 前向数据流 join 取交集（sroa.rs:241-260），load 无法解析到唯一到达值即整体放弃（:279-284）——「无 phi 所以不会错」的 claim 经核验成立 |
| licm | **grounded** | can_hoist（licm.rs:141-158）白名单排除 Div/Rem（可 fault）与 Fptosi（poison），不 hoist load/store/call/barrier；operands_available 显式要求操作数支配 preheader。speculation 安全论断真实成立 |
| cse | **principled** | 前向 must 可用性正确；但任何写 kill 全部 load（cse.rs:356）是保守近似——正确但粗于 LLVM 的 AA |
| loop_unroll | **principled** | 克隆保留 metadata，trip≤8 预算保守；无学术问题 |
| phi_alias | **principled** | is_safe_alias_source（phi_alias.rs:101-121）无显式支配检查，但 pred→target 边上 pred 内定义必 live，正确 |
| 其余（verify/canonicalize/loop_simplify/instcombine/dce+storage） | **principled** | 逐行核对无 soundness 缺口 |
| **链序本身** | **fitting** | 14-pass 顺序在代码、注释、git 历史（3b7cb89）与 PIPELINE_SPEC.md:50,73 中均无排序理由记录。SPEC 只写「链序不动」。这与 LLVM pipeline selector、GCC pass manager 的文献化排序形成对比。**最大的方法论弱点** |

### 映射族（sile-lsir/src/mapping.rs 五阶段）

| 阶段 | 裁决 | 依据与缺口 |
|---|---|---|
| policy | **半 grounded** | 线程数选择（min(plumbing_extent, tile_len, 256) 向下取 2 幂）合理；但 is_plumbing_header 靠 11 个硬编码块名前缀探测 plumbing 循环；shared_cap:16384 低于 WGSL spec 保证的 32KB |
| threads | **principled + 失真** | strip_loop 机制正确；但 doc 声称「同 Triton element ownership」不实——Triton sizePerThread 是每线程连续元素，此处是 cyclic 交错，偏离未记录；迭代独立性零结构检查（只有注释，:1441-1444） |
| reduce | **grounded + 一个真 bug** | 树归约内存序（scratch 写→barrier→log2(W) 步每步 barrier→copy-out→尾 barrier）正确。**真 bug**：`identity()`（mapping.rs:2800-2807）用 ±f32::MAX 作 Max/Min 单位元，而 lowering 丢弃 TileIR 的 identity 属性（lower/module.rs:3013 `let _ = (&identity, ...)`；TileIR 侧 cutile/emit.rs:2284 ReduceMax identity=-inf）——数据含 ±inf 时 Max/Min 归约错误。scratch_len 不查 shared_cap（运行时才拦） |
| mma | **principled** | flip_to_shared 的 barrier 交接（:2978）与均匀汇合 barrier（:3102）race-free 成立；但无使用面检查（其他读者未验证）、lane 循环不查 m*n%threads==0（reduce 查了，mma 没查——按 bug 补的检查，非原则） |
| cleanup | **fitting 倾向** | privatize_owner_tiles fragment 化扎实；但假 barrier（:1195-1220）与 OwnerOrdered 免检（:1130-1133）如总述 |

### 设计层叙事（相对文献的措辞）

- 「TTGIR 对应物已就位」**夸大**：无 per-value layout、无 convert_layout、无 swizzle、无布局推断、无 pipelining/warp specialization；policy 是每 target 全局常量。实际是 **WGSL 单端子集**，文档应降级措辞。
- LOWER_DESIGN.md 自相矛盾：§1 称「标量槽不进 SSA，连 phi 插入也省去」，§3 又由 ScalarPromote 插块参数；§16 引 Braun et al. CC 2013 论证方向反了——Braun 恰证明可以廉价维护 sealed-block。旧文未随实现同步。
- 「TileLang 只取一条启示」是半真：TileLang 的原语与 schedule 是一体两面，不是可单独摘取的一条。
- TileIR 冻结零 pass：**defensible**（真正理由是 diff=0 对齐契约，PIPELINE_SPEC.md:12，非学术主张），但代价已现：规范化压力堆进 cutile/emit.rs（118KB），tile 级优化无处安放。
- FP 树归约重结合：有记录有原则。**grounded**。

### 端到端正确性策略

- fail-closed 设计真实：WGSL 有 naga 真闭环；PTX 无 ptxas/in-tree 校验 = 空承诺（backend/lib.rs:54-59 自认）；`parallel_contract` 只查存在性配对，必要非充分。
- 静默降级（映射失败/contract 不过 → serial fallback，backend/lib.rs:56-67）让「fixture 通过」无法区分「并行正确」与「碰巧 serial」。有 SILE_SERIAL=1 调试开关可部分缓解。

## 一句话结论

**核心算法与内存模型是学术正统的；管线的组织方式（链序无理由、自指契约、装饰性校验、假 barrier）是工程拟合的。** 差距不在算法在方法论：三件最低成本修复——(1) 在 PIPELINE_SPEC 补 14-pass 链序的排序原则，(2) 修 identity 丢弃 bug（lowering 传递 identity 而非丢弃，mapping 消费之），(3) 把 LOWER_DESIGN §1/§3 矛盾与 TTGIR/TileLang 夸大措辞改掉。
