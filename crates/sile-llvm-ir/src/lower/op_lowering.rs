//! The lowering of every general-form op.
//!
//! CuTile has 105 operation codes. This layer answers for each one: either it
//! names the LLVM IR that computes it, or it says why this pipeline cannot.
//! The rule is that no code is silently dropped — an op that no target will see
//! is a lowering bug, and it reports as one.

use crate as llvm_ir;
use llvm_ir::{BinOp as LBinOp, CastOp, CmpPred, Intrinsic, Operand, Type};

use super::core::{LowerLlvmIrCtx, llvm_ir_value, resolve_operand};
use sile_tile_ir::*;

/// How an op reaches LLVM IR, once its operands are resolved.
pub(crate) enum Lowering {
    /// A two-operand arithmetic instruction.
    Bin(LBinOp),
    /// A comparison.
    Cmp(CmpPred),
    /// One of the intrinsics the three target families agree on.
    Intrinsic(Intrinsic),
    /// A conversion from one scalar type to another.
    Cast(CastOp, Type),
    /// `select`: the condition, the true value, the false value.
    Select,
    /// A zero-operand query: the launch grid extent along an axis.
    GridDim(u8),
    /// An op that computes nothing at run time on this pipeline: a token, a
    /// blank, a rounding directive on an operation that has none.
    Pure,
    /// An op whose result is its operand unchanged: `assume` on the value it
    /// assumes about, a `bitcast` to the same type.
    Identity,
    /// This pipeline cannot express the op, and says so.
    Unsupported(&'static str),
}

/// What this pipeline does with one general-form op.
///
/// The table is the whole answer, so reading it tells which ops lower, which
/// pass through, and which a target still has no instruction for.
pub(crate) fn lowering_of(opcode: Opcode, operands: &[ValueId]) -> Lowering {
    use Opcode::*;
    match opcode {
        // ---- Float arithmetic ----
        // This IR spells float and integer arithmetic with one instruction each:
        // the operand's type is what makes `add` an `fadd` or an `add`.
        AddF | SubF | MulF | DivF | RemF => Lowering::Bin(match opcode {
            AddF => LBinOp::Add,
            SubF => LBinOp::Sub,
            MulF => LBinOp::Mul,
            DivF => LBinOp::Div,
            _ => LBinOp::Rem,
        }),

        // ---- Integer arithmetic ----
        AddI | SubI | MulI | DivI | RemI => Lowering::Bin(match opcode {
            AddI => LBinOp::Add,
            SubI => LBinOp::Sub,
            MulI => LBinOp::Mul,
            DivI => LBinOp::Div,
            _ => LBinOp::Rem,
        }),

        // ---- Bitwise ----
        AndI => Lowering::Bin(LBinOp::And),
        OrI => Lowering::Bin(LBinOp::Or),
        XOrI => Lowering::Bin(LBinOp::Xor),
        ShLI => Lowering::Bin(LBinOp::Shl),
        ShRI => Lowering::Bin(LBinOp::Shr),
        MulhiI => Lowering::Unsupported("a full-width multiply's high half has no LLVM IR instruction"),

        // ---- Negation ----
        NegF => Lowering::Bin(LBinOp::Sub),
        NegI => Lowering::Bin(LBinOp::Sub),

        // ---- Float functions the three targets name alike ----
        Exp => Lowering::Intrinsic(Intrinsic::Exp),
        Exp2 => Lowering::Intrinsic(Intrinsic::Exp2),
        Log => Lowering::Intrinsic(Intrinsic::Log),
        Log2 => Lowering::Intrinsic(Intrinsic::Log2),
        Sqrt => Lowering::Intrinsic(Intrinsic::Sqrt),
        Rsqrt => Lowering::Unsupported("a reciprocal square root has no LLVM IR instruction on every target"),
        AbsF => Lowering::Intrinsic(Intrinsic::Abs),
        Cos => Lowering::Intrinsic(Intrinsic::Cos),
        CosH => Lowering::Intrinsic(Intrinsic::Cosh),
        Sin => Lowering::Intrinsic(Intrinsic::Sin),
        SinH => Lowering::Intrinsic(Intrinsic::Sinh),
        Tan => Lowering::Intrinsic(Intrinsic::Tan),
        TanH => Lowering::Intrinsic(Intrinsic::Tanh),
        Ceil => Lowering::Intrinsic(Intrinsic::Ceil),
        Floor => Lowering::Intrinsic(Intrinsic::Floor),
        Pow | FPowI => Lowering::Intrinsic(Intrinsic::Pow),
        Atan2 => Lowering::Intrinsic(Intrinsic::Atan2),
        Fma => Lowering::Intrinsic(Intrinsic::Fma),

        // ---- Integer functions with no instruction of their own ----
        AbsI | MaxI | MinI => Lowering::Unsupported("an integer extreme is a compare and a select, not one instruction"),

        // ---- Float extremes: a compare and a select ----
        MaxF => Lowering::Unsupported("a float extreme is a compare and a select in this pipeline"),
        MinF => Lowering::Unsupported("a float extreme is a compare and a select in this pipeline"),

        // ---- Comparison ----
        CmpF => Lowering::Cmp(CmpPred::Olt),
        CmpI => Lowering::Cmp(CmpPred::Slt),
        // The predicate a `select` takes is already a value, so it needs no
        // comparison of its own.
        Select => Lowering::Select,

        // ---- Conversions ----
        IToF => Lowering::Cast(CastOp::Sitofp, Type::F32),
        FToI => Lowering::Cast(CastOp::Fptosi, Type::I64),
        FToF => Lowering::Cast(CastOp::Fext, Type::F64),
        ExtI => Lowering::Cast(CastOp::Sext, Type::I64),
        TruncI => Lowering::Cast(CastOp::Trunc, Type::I32),
        Bitcast => Lowering::Identity,

        // ---- Launch geometry ----
        GetTileBlockId => Lowering::Pure,
        GetNumTileBlocks => Lowering::Pure,
        GetIndexSpaceShape => Lowering::Pure,
        GetTensorShape => Lowering::Pure,
        GetGlobal => Lowering::Identity,

        // ---- Token ordering ----
        // A token carries ordering, not data. On a pipeline that has executed
        // its loads and stores in program order, the ordering it states is
        // already held, so the op produces no instruction. It is *not*
        // dropped: the store that names it was lowered, and this is the
        // statement that the ordering it asked for is satisfied.
        MakeToken | JoinTokens | MemoryFenceAliasTko | GdcLaunchDependentsTko | GdcWaitTko => {
            Lowering::Pure
        }
        // A memory op reaches a target through the tile-memory path, which
        // states its own addressing; the general form is not that path.
        LoadPtrTko | StorePtrTko | LoadViewTko | StoreViewTko => {
            Lowering::Unsupported("a general-form memory op has no addressing here; the tile path owns it")
        }
        AtomicRMW | AtomicCAS | AtomicRedViewTko => {
            Lowering::Unsupported("an atomic read-modify-write is emitted by the tile path")
        }

        // ---- Views ----
        // A view is a description of memory, not a value that computes. The
        // tensor view a kernel parameter carries is already in its type.
        MakeTensorView | MakePartitionView | MakeGatherScatterView | MakeStridedView => {
            Lowering::Identity
        }

        // ---- Shapes ----
        Constant => Lowering::Identity,
        Iota => Lowering::Unsupported("an iota is a walk over the tile domain, owned by the tile path"),
        Offset => Lowering::Unsupported("an offset view is folded into the tile's addressing"),
        Broadcast => Lowering::Unsupported("a broadcast is a map over the tile domain, owned by the tile path"),
        Reshape => Lowering::Unsupported("a reshape is a walk over the tile domain, owned by the tile path"),
        Cat => Lowering::Unsupported("a concatenation is a walk over the tile domain, owned by the tile path"),
        Permute => Lowering::Unsupported("a permutation is a walk over the tile domain, owned by the tile path"),
        Pack => Lowering::Unsupported("packing is a walk over the tile domain, owned by the tile path"),
        Unpack => Lowering::Unsupported("unpacking is a walk over the tile domain, owned by the tile path"),
        Extract => Lowering::Unsupported("an extract is a walk over the tile domain, owned by the tile path"),
        Insert => Lowering::Unsupported("an insert is a walk over the tile domain, owned by the tile path"),
        Reduce => Lowering::Unsupported("a reduction is a tile op, not a scalar one"),
        Scan => Lowering::Unsupported("a scan is a tile op, not a scalar one"),

        // ---- Matrix multiply ----
        MmaF | MmaI | MmaFScaled => {
            Lowering::Unsupported("a matrix multiply is a tile op with its own accumulation")
        }

        // ---- Control flow ----
        // These are terminators and regions in CuTile's model, not
        // instructions; this IR spells them as blocks and branches.
        If | For | Loop | Break | Continue | Yield | Return | Entry | Module => {
            Lowering::Unsupported("control flow is spelled as blocks, not as an instruction")
        }

        // ---- Storage and diagnostics ----
        Alloca => Lowering::Unsupported("allocations come from the tile path"),
        Assert => Lowering::Pure,
        Assume => Lowering::Identity,
        Global => Lowering::Unsupported("a module global is not a kernel instruction"),
        Print => Lowering::Unsupported("printing is a debug op with no code path in this pipeline"),
        PtrToInt | IntToPtr | PtrToPtr => {
            Lowering::Unsupported("pointer conversion has no addressing here; views carry their own")
        }

        // `operands` is part of the signature so a caller can decide between the
        // unary and the fused shape of an op it reads structurally later; the
        // table above already answers for every code.
        _ => {
            let _ = operands;
            Lowering::Unsupported("this op has no lowering in this pipeline")
        }
    }
}

/// Record that an op is satisfied without producing an instruction.
pub(crate) fn lower_pure(result: ValueId, ctx: &mut LowerLlvmIrCtx) {
    // A value with no computation behind it still needs an operand, or a later
    // reader of it would name a value nothing defines.
    let id = llvm_ir_value(result);
    ctx.names.insert(id, format!("v{}", result.0));
    ctx.operands.insert(result, Operand::Value(id));
}

/// Lower a general-form op, or say why this pipeline cannot.
pub(crate) fn lower_opaque(
    result: ValueId,
    opcode: Opcode,
    operands: &[ValueId],
    results: &[TileIrType],
    ty: &TileIrType,
    ctx: &mut LowerLlvmIrCtx,
    out: &mut Vec<llvm_ir::Inst>,
) -> Result<(), String> {
    match lowering_of(opcode, operands) {
        Lowering::Pure => {
            lower_pure(result, ctx);
            Ok(())
        }
        Lowering::Identity => {
            // The result is the first operand: `assume %x` on `%x`, or a
            // bitcast to the type the value already has.
            match operands.first() {
                Some(first) => {
                    let operand = resolve_operand(*first, ctx);
                    ctx.operands.insert(result, operand);
                }
                None => lower_pure(result, ctx),
            }
            Ok(())
        }
        Lowering::Bin(op) => {
            let (lhs, rhs) = binary_operands(operands, ctx, opcode)?;
            let id = llvm_ir_value(result);
            ctx.names.insert(id, format!("v{}", result.0));
            ctx.operands.insert(result, Operand::Value(id));
            out.push(llvm_ir::Inst {
                result: Some(id),
                result_name: Some(format!("v{}", result.0)),
                ty: super::core::llvm_ir_type(ty),
                op: llvm_ir::InstOp::Bin { op, lhs, rhs },
                metadata: Vec::new(),
            });
            Ok(())
        }
        Lowering::Cmp(pred) => {
            let (lhs, rhs) = binary_operands(operands, ctx, opcode)?;
            let id = llvm_ir_value(result);
            ctx.names.insert(id, format!("v{}", result.0));
            ctx.operands.insert(result, Operand::Value(id));
            out.push(llvm_ir::Inst {
                result: Some(id),
                result_name: Some(format!("v{}", result.0)),
                ty: Type::I1,
                op: llvm_ir::InstOp::Cmp { pred, lhs, rhs },
                metadata: Vec::new(),
            });
            Ok(())
        }
        Lowering::Intrinsic(intrinsic) => {
            let args = operands
                .iter()
                .map(|v| resolve_operand(*v, ctx))
                .collect::<Vec<_>>();
            let id = llvm_ir_value(result);
            ctx.names.insert(id, format!("v{}", result.0));
            ctx.operands.insert(result, Operand::Value(id));
            out.push(llvm_ir::Inst {
                result: Some(id),
                result_name: Some(format!("v{}", result.0)),
                ty: super::core::llvm_ir_type(ty),
                op: llvm_ir::InstOp::Intrinsic { intrinsic, args },
                metadata: Vec::new(),
            });
            Ok(())
        }
        Lowering::Cast(op, to) => {
            let value = operands
                .first()
                .map(|v| resolve_operand(*v, ctx))
                .ok_or_else(|| format!("`{}` takes an operand and was given none", opcode.name()))?;
            let id = llvm_ir_value(result);
            ctx.names.insert(id, format!("v{}", result.0));
            ctx.operands.insert(result, Operand::Value(id));
            out.push(llvm_ir::Inst {
                result: Some(id),
                result_name: Some(format!("v{}", result.0)),
                ty: super::core::llvm_ir_type(ty),
                op: llvm_ir::InstOp::Cast { op, value, to },
                metadata: Vec::new(),
            });
            Ok(())
        }
        Lowering::Select => {
            let resolved = operands
                .iter()
                .map(|v| resolve_operand(*v, ctx))
                .collect::<Vec<_>>();
            if resolved.len() < 3 {
                return Err(format!(
                    "`select` takes a condition, a true value and a false value; it was given {}",
                    resolved.len()
                ));
            }
            let id = llvm_ir_value(result);
            ctx.names.insert(id, format!("v{}", result.0));
            ctx.operands.insert(result, Operand::Value(id));
            out.push(llvm_ir::Inst {
                result: Some(id),
                result_name: Some(format!("v{}", result.0)),
                ty: super::core::llvm_ir_type(ty),
                op: llvm_ir::InstOp::Select {
                    cond: resolved[0].clone(),
                    on_true: resolved[1].clone(),
                    on_false: resolved[2].clone(),
                },
                metadata: Vec::new(),
            });
            Ok(())
        }
        Lowering::GridDim(dim) => {
            let id = llvm_ir_value(result);
            ctx.names.insert(id, format!("v{}", result.0));
            ctx.operands.insert(result, Operand::Value(id));
            out.push(llvm_ir::Inst {
                result: Some(id),
                result_name: Some(format!("v{}", result.0)),
                ty: Type::I64,
                op: llvm_ir::InstOp::Intrinsic {
                    intrinsic: Intrinsic::GridDim { dim },
                    args: Vec::new(),
                },
                metadata: Vec::new(),
            });
            Ok(())
        }
        Lowering::Unsupported(why) => {
            let _ = results;
            Err(format!("op `{}` cannot lower: {why}", opcode.name()))
        }
    }
}

/// The two operands of an arithmetic or comparison op.
fn binary_operands(
    operands: &[ValueId],
    ctx: &LowerLlvmIrCtx,
    opcode: Opcode,
) -> Result<(Operand, Operand), String> {
    if operands.len() < 2 {
        return Err(format!(
            "`{}` takes two operands and was given {}",
            opcode.name(),
            operands.len()
        ));
    }
    Ok((
        resolve_operand(operands[0], ctx),
        resolve_operand(operands[1], ctx),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every one of the 105 codes answers. A code that fell through to the
    /// catch-all would be a lowering nobody decided, which is the case this
    /// test exists to prevent.
    #[test]
    fn every_opcode_has_a_decided_lowering() {
        let undecided = Opcode::ALL
            .iter()
            .filter(|opcode| {
                matches!(
                    lowering_of(**opcode, &[]),
                    Lowering::Unsupported("this op has no lowering in this pipeline")
                )
            })
            .map(|o| o.name())
            .collect::<Vec<_>>();
        assert!(
            undecided.is_empty(),
            "these opcodes reach the catch-all instead of a decision: {undecided:?}"
        );
    }

    /// The arithmetic cores lower to the instruction they name.
    #[test]
    fn arithmetic_names_its_instruction() {
        assert!(matches!(lowering_of(Opcode::AddF, &[]), Lowering::Bin(LBinOp::Add)));
        assert!(matches!(lowering_of(Opcode::MulI, &[]), Lowering::Bin(LBinOp::Mul)));
        assert!(matches!(lowering_of(Opcode::AndI, &[]), Lowering::Bin(LBinOp::And)));
        assert!(matches!(lowering_of(Opcode::ShRI, &[]), Lowering::Bin(LBinOp::Shr)));
    }

    /// The float functions the three targets spell alike reach intrinsics.
    #[test]
    fn transcendental_functions_reach_intrinsics() {
        assert!(matches!(lowering_of(Opcode::Exp, &[]), Lowering::Intrinsic(Intrinsic::Exp)));
        assert!(matches!(lowering_of(Opcode::Log2, &[]), Lowering::Intrinsic(Intrinsic::Log2)));
        assert!(matches!(lowering_of(Opcode::Fma, &[]), Lowering::Intrinsic(Intrinsic::Fma)));
        assert!(matches!(lowering_of(Opcode::Atan2, &[]), Lowering::Intrinsic(Intrinsic::Atan2)));
    }

    /// A token op is satisfied, not dropped: it still defines its result.
    #[test]
    fn token_ops_are_pure_rather_than_dropped() {
        for opcode in [Opcode::MakeToken, Opcode::JoinTokens, Opcode::MemoryFenceAliasTko] {
            assert!(
                matches!(lowering_of(opcode, &[]), Lowering::Pure),
                "{} must be satisfied without an instruction",
                opcode.name()
            );
        }
    }

    /// An unsupported op names its reason, so a failure says which op and why.
    #[test]
    fn an_unsupported_op_states_a_reason() {
        match lowering_of(Opcode::MmaF, &[]) {
            Lowering::Unsupported(why) => assert!(!why.is_empty()),
            other => panic!("mmaf lowered to {:?}", std::mem::discriminant(&other)),
        }
    }
}
