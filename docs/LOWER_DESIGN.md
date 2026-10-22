# TileIR → LSIR 下降设计

本文界定 sile 编译器中段的两级契约：TileIR（结构化）下降为 LSIR（CFG/SSA），LSIR 发射为四后端（C/WGSL/PTX/Metal）。所有实现以本文为准；实现与本文冲突时，改实现。

## 1. 理论定位

- TileIR 是结构化 tile 语言：区域嵌套，`for`/`if` 是 SESE 区域（single-entry, single-exit）。类比 MLIR 的 `scf` 方言。
- LSIR 是任意 CFG 上的 pruned SSA。
- 结构化语言到 CFG 的下降是**保序模板变换**：每个区域算子映射到一个固定 CFG 模板。`for` → header/body/exit 三角；`if` → then/else/join 菱形；`yield` → join 前的拷贝。
- SSA 构造是**独立阶段**，不是下降的一部分。依据 Cytron–Ferrante–Rosen–Wegman–Zadeck (TOPLAS 1991)：先把一切标量落在 alloca（stack IR，dominance 平凡成立），再按 dominance frontier 插 phi 并重命名。这就是 LLVM `mem2reg`。本管线只取其前半：SROA 对常数索引的数组槽做无 phi 的提升，标量槽在下降阶段不进 SSA，留待管线的 ScalarPromote（§3）进入块参数形 SSA。

推论：**下降阶段禁止手写 phi 与块参数**。 Braun et al. (CC 2013) 证明从 SSA 按需构造可以廉价维护 sealed-block 不变量，但那是有框架支撑的逐点构造；本下降器是逐算子的保序模板发射器，没有那个框架，历史上手写 phi 的下降器在 `materialize`/`for`/`mma`/`if` 各处反复产生 dominance 破坏，逐点修补不可收敛。零 phi 下沉把 dominance 正确性从逐发射点的不变量变成 alloca 即入口的构造性质（INV3），破坏面为零。

## 2. 下降器四条不变量

INV1 **唯一插入点**。`self.current` 永远是"下一条指令该落的块"。任何函数返回时，`current` 停在本函数指令流的续接块（无 terminator）。禁止在设置 terminator 之后向同一块继续发射。

INV2 **跨块先物化**。惰性 recipe 只允许在建立它的块内折叠。跨块边界（循环体、分支臂、join）之前，一切待用值必须物化为 alloca。`counted_loop`/`lower_for`/`lower_if` 的序言各调用一次 `materialize_all`，且必须在创建 header 之前——物化环是发射中的一部分，它链在入口边之前。

INV3 **alloca 即作用域**。值的一切存储是入口块的 alloca。定义支配使用由构造平凡保证：读写在任意块发生，alloca 定义在入口支配全图。

INV4 **下降零 phi**。下降器的输出不含 phi 与块参数。循环变量与 carry 值是 alloca：入口 store 初值，header load 判条件，回边 store 步进，exit load 逸出值。

## 3. 阶段与 pass 职责

```
TileIR
  │ 下降（本设计 §2）          —— 结构保序，零 phi，全 alloca
  ▼
stack LSIR
  │ VerifyInput               —— 结构合法
  │ Canonicalize              —— 消除平凡形式
  │ LoopSimplify              —— preheader / 单回边
  │ SROA                      —— tile 数组 alloca 的可提升子集
  │ ScalarPromote             —— 私有标量槽 → 块参数（mem2reg 的标量半边）
  │ CSE / InstCombine / LICM / LoopUnroll / DCE
  │ VerifyOutput              —— dominance 与 SSA 终验
  ▼
SSA LSIR
  │ emit（模式匹配，见 §4）
  ▼
C / WGSL / PTX / Metal
```

- **零 phi 契约只管下降器，不管 pass**：`INV4` 约束的是下降输出，不是管线终态。`ScalarPromote`（Sroa 之后、Dce 之前）把私有标量槽提升为块参数——phi 的块参数形——正是把零 phi 的代价清掉的那一步。放置用迭代支配边界，按活跃性剪枝（槽在某块未赋值时不插参数），再沿支配树重命名。支配与支配边界来自 `passes/cfg.rs`。
- **SROA**（已有）：把常数索引的 tile alloca 拆量并提升。unroll 之后收益最大。它只处理数组槽；标量槽归 `ScalarPromote`。
- **PhiAlias**（保留）：下降器不产生 phi；此 pass 折叠万一出现的平凡 phi（单前驱、同值），是零 phi 契约的安全网。
- **emit 认两种计数循环形态**：alloca 形（header load / 回边 store）与提升后的块参数形（header 参数 / 边参数）。两者都拼 `for` 并保留向量化提示，`verify_structured_shapes` 同时接受。

## 4. 后端 emit 契约

C/WGSL/Metal 是结构化语言，不接受任意 CFG。通用解是 CFG 结构化恢复（relooper/stackifier）；我们不需要——下降器由我们控制，emit 只面对**三种规范形状**，逐形状模板翻译：

| 形状 | LLVM 模式 | C/Metal/WGSL 输出 |
|---|---|---|
| 直线 | 无回边的块链 | 语句序列 |
| 计数循环 | header(load,cmp,condbr) / body / 回边 store / exit | `for (i = 0; i < n; ++i)` |
| 选择 | condbr + then/else + join | `if (c) {…} else {…}`（join 值为 alloca） |

emit 侧的义务：按形状匹配；匹配失败即报错，不得猜测。PTX 是线性分支目标，CFG 直译，无此契约。

verify 增设形状检查（可选，M2 落）：判定输入符合三形状文法，把"emit 猜错形状"提前为编译错误。

## 5. 验收

- 八个数值示例（saxpy/softmax/rms_norm/gemm/gemm_static/batch_matmul/flash_attention/flash_attention_causal）C target 全绿。
- `scripts/check_tileir_alignment.sh` 17/17。
- 下降器输出经 `lsir-00-lowered` 抽查：无 `phi`。
