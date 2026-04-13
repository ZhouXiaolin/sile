use std::collections::{HashMap, HashSet};

use sile_core::{Error, Result};
use sile_llvm_ir as llvm_ir;
use sile_llvm_ir::Function as LlvmIrFunction;

use crate::cpu::codegen::generate as generate_llvm_ir_c_source;
use crate::metal::codegen::generate as generate_llvm_ir_metal_source;
use crate::{BackendArtifact, CodegenTarget};

pub fn run(llvm_ir: &LlvmIrFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    let artifact = match target {
        CodegenTarget::C => BackendArtifact::CSource(generate_llvm_ir_c_source(llvm_ir)?),
        CodegenTarget::Metal => {
            BackendArtifact::MetalSource(generate_llvm_ir_metal_source(llvm_ir)?)
        }
    };
    Ok(artifact)
}

pub fn build_value_names(func: &llvm_ir::Function) -> HashMap<llvm_ir::ValueId, String> {
    let mut names = HashMap::new();
    for param in &func.params {
        names.insert(param.id, param.name.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            names.insert(param.id, param.name.clone());
        }
        for inst in &block.insts {
            if let (Some(id), Some(name)) = (inst.result, inst.result_name.as_ref()) {
                names.insert(id, name.clone());
            }
        }
    }
    names
}

pub fn value_name(value_names: &HashMap<llvm_ir::ValueId, String>, id: llvm_ir::ValueId) -> String {
    value_names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("v{}", id.0))
}

pub fn format_operand(
    value_names: &HashMap<llvm_ir::ValueId, String>,
    operand: &llvm_ir::Operand,
) -> String {
    match operand {
        llvm_ir::Operand::Value(id) => value_name(value_names, *id),
        llvm_ir::Operand::Const(llvm_ir::Constant::Int(value)) => value.to_string(),
        llvm_ir::Operand::Const(llvm_ir::Constant::Float(value)) => float_literal(*value),
        llvm_ir::Operand::Const(llvm_ir::Constant::Bool(value)) => {
            if *value {
                "true".into()
            } else {
                "false".into()
            }
        }
    }
}

pub fn array_dims(ty: &llvm_ir::Type) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut current = ty;
    while let llvm_ir::Type::Array { len, elem } = current {
        dims.push(*len);
        current = elem;
    }
    dims
}

pub fn bin_op_symbol(op: llvm_ir::BinOp) -> &'static str {
    match op {
        llvm_ir::BinOp::Add => "+",
        llvm_ir::BinOp::Sub => "-",
        llvm_ir::BinOp::Mul => "*",
        llvm_ir::BinOp::Div => "/",
        llvm_ir::BinOp::And => "&",
        llvm_ir::BinOp::Or => "|",
    }
}

pub fn cmp_pred_symbol(pred: llvm_ir::CmpPred) -> &'static str {
    match pred {
        llvm_ir::CmpPred::Eq => "==",
        llvm_ir::CmpPred::Ne => "!=",
        llvm_ir::CmpPred::Slt | llvm_ir::CmpPred::Olt => "<",
        llvm_ir::CmpPred::Sle | llvm_ir::CmpPred::Ole => "<=",
        llvm_ir::CmpPred::Sgt | llvm_ir::CmpPred::Ogt => ">",
        llvm_ir::CmpPred::Sge | llvm_ir::CmpPred::Oge => ">=",
    }
}

pub fn float_literal(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}f")
    } else {
        format!("{value}f")
    }
}

pub fn lower_common_inst_line<FValue, FOperand>(
    inst: &llvm_ir::Inst,
    value_name: FValue,
    format_operand: FOperand,
) -> Option<String>
where
    FValue: Fn(llvm_ir::ValueId) -> String,
    FOperand: Fn(&llvm_ir::Operand) -> String,
{
    match &inst.op {
        llvm_ir::InstOp::Bin { op, lhs, rhs } => inst.result.map(|id| {
            format!(
                "{} = {} {} {};",
                value_name(id),
                format_operand(lhs),
                bin_op_symbol(*op),
                format_operand(rhs)
            )
        }),
        llvm_ir::InstOp::Cmp { pred, lhs, rhs } => inst.result.map(|id| {
            format!(
                "{} = {} {} {};",
                value_name(id),
                format_operand(lhs),
                cmp_pred_symbol(*pred),
                format_operand(rhs)
            )
        }),
        llvm_ir::InstOp::Select {
            cond,
            on_true,
            on_false,
        } => inst.result.map(|id| {
            format!(
                "{} = ({}) ? ({}) : ({});",
                value_name(id),
                format_operand(cond),
                format_operand(on_true),
                format_operand(on_false)
            )
        }),
        llvm_ir::InstOp::Load { ptr } => inst
            .result
            .map(|id| format!("{} = *({});", value_name(id), format_operand(ptr))),
        llvm_ir::InstOp::Store { ptr, value } => Some(format!(
            "*({}) = {};",
            format_operand(ptr),
            format_operand(value)
        )),
        llvm_ir::InstOp::Cast { value, .. } => inst
            .result
            .map(|id| format!("{} = {};", value_name(id), format_operand(value))),
        llvm_ir::InstOp::Gep { base, indices } => inst.result.map(|id| {
            let index_suffix = indices
                .iter()
                .map(|idx| format!("[{}]", format_operand(idx)))
                .collect::<Vec<_>>()
                .join("");
            format!(
                "{} = &({}{});",
                value_name(id),
                format_operand(base),
                index_suffix
            )
        }),
        _ => None,
    }
}

pub trait TextCodegen: Sized {
    fn emit_prelude(&mut self);
    fn emit_signature(&mut self);
    fn emit_body(&mut self) -> Result<()>;
    fn finish(self) -> String;
}

pub fn generate_text<E: TextCodegen>(mut emitter: E) -> Result<String> {
    emitter.emit_prelude();
    emitter.emit_signature();
    emitter.emit_body()?;
    Ok(emitter.finish())
}

pub fn emit_value_decls<F>(func: &llvm_ir::Function, mut emit_decl: F) -> bool
where
    F: FnMut(llvm_ir::ValueId, &llvm_ir::Type),
{
    let mut declared = HashMap::new();

    for block in &func.blocks {
        for param in &block.params {
            declared.entry(param.id).or_insert(&param.ty);
        }
        for inst in &block.insts {
            if let Some(id) = inst.result {
                declared.entry(id).or_insert(&inst.ty);
            }
        }
    }

    if declared.is_empty() {
        return false;
    }

    for block in &func.blocks {
        for param in &block.params {
            if declared.remove(&param.id).is_some() {
                emit_decl(param.id, &param.ty);
            }
        }
        for inst in &block.insts {
            if let Some(id) = inst.result
                && let Some(ty) = declared.remove(&id)
            {
                emit_decl(id, ty);
            }
        }
    }

    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePlan {
    pub output_param: usize,
    pub rows: usize,
    pub cols: usize,
}

pub fn infer_tile_plan(func: &llvm_ir::Function) -> Option<TilePlan> {
    let inst_by_result = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect::<HashMap<_, _>>();
    let value_types = build_value_type_map(func);

    for block in &func.blocks {
        for inst in &block.insts {
            let llvm_ir::InstOp::Store {
                ptr: llvm_ir::Operand::Value(ptr_id),
                value: llvm_ir::Operand::Value(value_id),
            } = &inst.op
            else {
                continue;
            };
            let Some(ptr_inst) = inst_by_result.get(ptr_id) else {
                continue;
            };
            let llvm_ir::InstOp::Gep {
                base: llvm_ir::Operand::Value(buf_id),
                ..
            } = &ptr_inst.op
            else {
                continue;
            };
            let Some(output_param) = func.params.iter().position(|param| param.id == *buf_id)
            else {
                continue;
            };
            let Some((rows, cols)) =
                infer_tile_dims_from_scalar_store(*value_id, &inst_by_result, &value_types)
            else {
                let Some((rows, cols)) =
                    infer_tile_dims_from_direct_loop_store(func, block, &inst_by_result)
                else {
                    continue;
                };
                return Some(TilePlan {
                    output_param,
                    rows,
                    cols,
                });
            };
            return Some(TilePlan {
                output_param,
                rows,
                cols,
            });
        }
    }

    None
}

pub trait StructuredEmitter {
    fn emit_block_insts(&mut self, block: &llvm_ir::BasicBlock) -> Result<()>;
    fn emit_block_insts_except(
        &mut self,
        block: &llvm_ir::BasicBlock,
        skip_result: Option<llvm_ir::ValueId>,
    ) -> Result<()>;
    fn emit_block_param_assignments_skipping(
        &mut self,
        target: llvm_ir::BlockId,
        args: &[llvm_ir::Operand],
        skip_params: &[llvm_ir::ValueId],
    );
    fn emit_block_param_assignments(&mut self, target: llvm_ir::BlockId, args: &[llvm_ir::Operand]) {
        self.emit_block_param_assignments_skipping(target, args, &[]);
    }
    fn format_operand(&self, operand: &llvm_ir::Operand) -> String;
    fn writeln(&mut self, line: &str);
    fn indent_inc(&mut self);
    fn indent_dec(&mut self);
}

#[derive(Clone, Copy)]
pub struct StructuredCfgMessages {
    pub preheader_must_branch: &'static str,
    pub missing_loop_header: &'static str,
    pub header_must_cond_br: &'static str,
    pub loop_backedge_mismatch: &'static str,
    pub unsupported_cond_br: &'static str,
    pub unsupported_switch: &'static str,
}

pub fn emit_structured_from<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    start: llvm_ir::BlockId,
    stop_targets: &[llvm_ir::BlockId],
    stop_skip_params: Option<(llvm_ir::BlockId, Vec<llvm_ir::ValueId>)>,
    messages: StructuredCfgMessages,
) -> Result<Option<llvm_ir::BlockId>> {
    let mut current = start;
    loop {
        let block = get_block(func, current)
            .cloned()
            .ok_or_else(|| Error::Compile(format!("missing LLVM IR block {:?}", current)))?;

        if let llvm_ir::Terminator::Br { target, args } = &block.terminator
            && stop_targets.contains(target)
        {
            emitter.emit_block_insts(&block)?;
            emitter.emit_block_param_assignments_skipping(
                *target,
                args,
                skip_param_ids_for_target(stop_skip_params.as_ref(), *target),
            );
            return Ok(Some(*target));
        }

        if is_loop_preheader(func, &block) {
            current = emit_structured_loop_preheader(emitter, func, &block, messages)?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        if is_loop_header(func, &block) {
            current = emit_structured_loop_header(emitter, func, &block, messages)?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        if matches!(block.terminator, llvm_ir::Terminator::CondBr { .. }) {
            current = emit_structured_if_else(emitter, func, &block, messages)?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        emitter.emit_block_insts(&block)?;
        match &block.terminator {
            llvm_ir::Terminator::Br { target, args } => {
                emitter.emit_block_param_assignments_skipping(
                    *target,
                    args,
                    skip_param_ids_for_target(stop_skip_params.as_ref(), *target),
                );
                if stop_targets.contains(target) {
                    return Ok(Some(*target));
                }
                current = *target;
            }
            llvm_ir::Terminator::Ret { value: None } => {
                emitter.writeln("return;");
                return Ok(None);
            }
            llvm_ir::Terminator::Ret { value: Some(value) } => {
                emitter.writeln(&format!("return {};", emitter.format_operand(value)));
                return Ok(None);
            }
            llvm_ir::Terminator::CondBr { .. } => {
                return Err(Error::Compile(messages.unsupported_cond_br.into()));
            }
            llvm_ir::Terminator::Switch { .. } => {
                return Err(Error::Compile(messages.unsupported_switch.into()));
            }
        }
    }
}

fn emit_structured_loop_preheader<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    preheader: &llvm_ir::BasicBlock,
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let llvm_ir::Terminator::Br {
        target: header_id,
        args: header_args,
    } = &preheader.terminator
    else {
        return Err(Error::Compile(messages.preheader_must_branch.into()));
    };
    let header = get_block(func, *header_id)
        .cloned()
        .ok_or_else(|| Error::Compile(messages.missing_loop_header.into()))?;
    let llvm_ir::Terminator::CondBr {
        cond,
        true_target,
        true_args,
        false_target,
        false_args,
    } = &header.terminator
    else {
        return Err(Error::Compile(messages.header_must_cond_br.into()));
    };

    emitter.emit_block_insts(preheader)?;
    emit_structured_loop_header_impl(
        emitter,
        func,
        &header,
        Some(header_args),
        cond,
        *true_target,
        true_args,
        *false_target,
        false_args,
        messages,
    )
}

fn emit_structured_loop_header<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    header: &llvm_ir::BasicBlock,
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let llvm_ir::Terminator::CondBr {
        cond,
        true_target,
        true_args,
        false_target,
        false_args,
    } = &header.terminator
    else {
        return Err(Error::Compile(messages.header_must_cond_br.into()));
    };
    emit_structured_loop_header_impl(
        emitter,
        func,
        header,
        None,
        cond,
        *true_target,
        true_args,
        *false_target,
        false_args,
        messages,
    )
}

fn emit_structured_if_else<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    block: &llvm_ir::BasicBlock,
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let llvm_ir::Terminator::CondBr {
        cond,
        true_target,
        true_args,
        false_target,
        false_args,
    } = &block.terminator
    else {
        return Err(Error::Compile(messages.unsupported_cond_br.into()));
    };
    let join = shared_branch_join(func, *true_target, *false_target)
        .ok_or_else(|| Error::Compile(messages.unsupported_cond_br.into()))?;

    let (cond_expr, skip_result) = branch_condition_expr(func, block, cond, false);
    emitter.emit_block_insts_except(block, skip_result)?;
    emitter.writeln(&format!("if ({cond_expr}) {{"));
    emitter.indent_inc();
    emitter.emit_block_param_assignments(*true_target, true_args);
    if emit_structured_from(emitter, func, *true_target, &[join], None, messages)? != Some(join) {
        return Err(Error::Compile(messages.unsupported_cond_br.into()));
    }
    emitter.indent_dec();
    emitter.writeln("} else {");
    emitter.indent_inc();
    emitter.emit_block_param_assignments(*false_target, false_args);
    if emit_structured_from(emitter, func, *false_target, &[join], None, messages)? != Some(join) {
        return Err(Error::Compile(messages.unsupported_cond_br.into()));
    }
    emitter.indent_dec();
    emitter.writeln("}");

    Ok(join)
}

#[allow(clippy::too_many_arguments)]
fn emit_structured_loop_header_impl<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llvm_ir::Function,
    header: &llvm_ir::BasicBlock,
    incoming_header_args: Option<&[llvm_ir::Operand]>,
    cond: &llvm_ir::Operand,
    true_target: llvm_ir::BlockId,
    true_args: &[llvm_ir::Operand],
    false_target: llvm_ir::BlockId,
    false_args: &[llvm_ir::Operand],
    messages: StructuredCfgMessages,
) -> Result<llvm_ir::BlockId> {
    let optimized_loop = branch_condition_expr(func, header, cond, true);
    match optimized_loop {
        (cond_expr, skip_result @ Some(_))
            if header_only_contains_skipped_cmp(header, skip_result) =>
        {
            if let Some(for_loop) =
                detect_for_loop_induction(func, header, cond, true_target, true_args)
            {
                let induction_name =
                    emitter.format_operand(&llvm_ir::Operand::Value(for_loop.induction_param));
                if let Some(args) = incoming_header_args {
                    emitter.emit_block_param_assignments_skipping(
                        header.id,
                        args,
                        &[for_loop.induction_param],
                    );
                }
                let init_expr = incoming_header_args
                    .and_then(|args| args.get(for_loop.induction_param_index))
                    .map(|arg| format!("{induction_name} = {}", emitter.format_operand(arg)))
                    .unwrap_or_default();
                let step_expr = induction_step_expr(&induction_name, for_loop.step);
                if !init_expr.is_empty() && is_simple_loop_body(func, true_target, header.id) {
                    emitter.writeln("#pragma omp simd");
                }
                emitter.writeln(&format!(
                    "for ({}; {}; {}) {{",
                    init_expr, cond_expr, step_expr
                ));
                emitter.indent_inc();
                emitter.emit_block_param_assignments(true_target, true_args);
                let backedge = emit_structured_from(
                    emitter,
                    func,
                    true_target,
                    &[header.id],
                    Some((header.id, vec![for_loop.induction_param])),
                    messages,
                )?;
                if backedge != Some(header.id) {
                    return Err(Error::Compile(messages.loop_backedge_mismatch.into()));
                }
                emitter.indent_dec();
                emitter.writeln("}");
                emitter.emit_block_param_assignments(false_target, false_args);
                return Ok(false_target);
            }

            if let Some(args) = incoming_header_args {
                emitter.emit_block_param_assignments(header.id, args);
            }
            emitter.writeln(&format!("while ({cond_expr}) {{"));
            emitter.indent_inc();
            emitter.emit_block_param_assignments(true_target, true_args);
            let backedge = emit_structured_from(
                emitter,
                func,
                true_target,
                &[header.id],
                None,
                messages,
            )?;
            if backedge != Some(header.id) {
                return Err(Error::Compile(messages.loop_backedge_mismatch.into()));
            }
            emitter.indent_dec();
            emitter.writeln("}");
            emitter.emit_block_param_assignments(false_target, false_args);
            Ok(false_target)
        }
        _ => {
            if let Some(args) = incoming_header_args {
                emitter.emit_block_param_assignments(header.id, args);
            }
            emitter.writeln("while (true) {");
            emitter.indent_inc();
            emitter.emit_block_insts(header)?;
            emitter.writeln(&format!("if (!({})) {{", emitter.format_operand(cond)));
            emitter.indent_inc();
            emitter.emit_block_param_assignments(false_target, false_args);
            emitter.writeln("break;");
            emitter.indent_dec();
            emitter.writeln("}");

            emitter.emit_block_param_assignments(true_target, true_args);
            let backedge = emit_structured_from(
                emitter,
                func,
                true_target,
                &[header.id],
                None,
                messages,
            )?;
            if backedge != Some(header.id) {
                return Err(Error::Compile(messages.loop_backedge_mismatch.into()));
            }
            emitter.indent_dec();
            emitter.writeln("}");

            Ok(false_target)
        }
    }
}

#[derive(Clone, Copy)]
struct ForLoopInduction {
    induction_param: llvm_ir::ValueId,
    induction_param_index: usize,
    step: i64,
}

fn detect_for_loop_induction(
    func: &llvm_ir::Function,
    header: &llvm_ir::BasicBlock,
    cond: &llvm_ir::Operand,
    true_target: llvm_ir::BlockId,
    _true_args: &[llvm_ir::Operand],
) -> Option<ForLoopInduction> {
    let cond_id = operand_value_id(cond)?;
    let cmp_inst = get_inst_by_result(func, cond_id)?;
    let llvm_ir::InstOp::Cmp { pred, lhs, rhs } = &cmp_inst.op else {
        return None;
    };
    let llvm_ir::Operand::Value(loop_param) = lhs else {
        return None;
    };
    if !matches!(
        pred,
        llvm_ir::CmpPred::Slt | llvm_ir::CmpPred::Sle | llvm_ir::CmpPred::Sgt | llvm_ir::CmpPred::Sge
    ) {
        return None;
    }
    let induction_param_index = header.params.iter().position(|param| param.id == *loop_param)?;
    let backedge_candidates = find_loop_backedge_args(func, header.id, true_target);
    let step = backedge_candidates.into_iter().find_map(|args| {
        induction_step_from_backedge_arg(func, *loop_param, args.get(induction_param_index)?)
    })?;
    if step == 0 {
        return None;
    }
    if !matches!(
        (step > 0, pred),
        (true, llvm_ir::CmpPred::Slt | llvm_ir::CmpPred::Sle)
            | (false, llvm_ir::CmpPred::Sgt | llvm_ir::CmpPred::Sge)
    ) {
        return None;
    }
    if !matches!(rhs, llvm_ir::Operand::Const(_) | llvm_ir::Operand::Value(_)) {
        return None;
    }
    Some(ForLoopInduction {
        induction_param: *loop_param,
        induction_param_index,
        step,
    })
}

fn find_loop_backedge_args(
    func: &llvm_ir::Function,
    header_id: llvm_ir::BlockId,
    true_target: llvm_ir::BlockId,
) -> Vec<Vec<llvm_ir::Operand>> {
    func.blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            llvm_ir::Terminator::Br { target, args }
                if *target == header_id
                    && block_reaches(func, true_target, block.id, &mut HashSet::new()) =>
            {
                Some(args.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
}

fn induction_step_from_backedge_arg(
    func: &llvm_ir::Function,
    loop_param: llvm_ir::ValueId,
    arg: &llvm_ir::Operand,
) -> Option<i64> {
    let llvm_ir::Operand::Value(step_value_id) = arg else {
        return None;
    };
    let step_inst = get_inst_by_result(func, *step_value_id)?;
    let llvm_ir::InstOp::Bin { op, lhs, rhs } = &step_inst.op else {
        return None;
    };
    match (op, lhs, rhs) {
        (
            llvm_ir::BinOp::Add,
            llvm_ir::Operand::Value(id),
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(step)),
        ) if *id == loop_param => Some(*step),
        (
            llvm_ir::BinOp::Add,
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(step)),
            llvm_ir::Operand::Value(id),
        ) if *id == loop_param => Some(*step),
        (
            llvm_ir::BinOp::Sub,
            llvm_ir::Operand::Value(id),
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(step)),
        ) if *id == loop_param => Some(-*step),
        _ => None,
    }
}

fn induction_step_expr(name: &str, step: i64) -> String {
    match step {
        1 => format!("++{name}"),
        -1 => format!("--{name}"),
        value if value > 0 => format!("{name} += {value}"),
        value => format!("{name} -= {}", -value),
    }
}

fn is_simple_loop_body(
    func: &llvm_ir::Function,
    body_start: llvm_ir::BlockId,
    header_id: llvm_ir::BlockId,
) -> bool {
    let mut current = body_start;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(current) {
            return false;
        }
        let Some(block) = get_block(func, current) else {
            return false;
        };
        for inst in &block.insts {
            if matches!(
                &inst.op,
                llvm_ir::InstOp::Call { .. }
                    | llvm_ir::InstOp::Intrinsic {
                        intrinsic: llvm_ir::Intrinsic::Exp,
                        ..
                    }
            ) {
                return false;
            }
        }
        match &block.terminator {
            llvm_ir::Terminator::Br { target, .. } if *target == header_id => return true,
            llvm_ir::Terminator::Br { target, .. } => current = *target,
            _ => return false,
        }
    }
}

fn skip_param_ids_for_target<'a>(
    skip: Option<&'a (llvm_ir::BlockId, Vec<llvm_ir::ValueId>)>,
    target: llvm_ir::BlockId,
) -> &'a [llvm_ir::ValueId] {
    if let Some((skip_target, ids)) = skip
        && *skip_target == target
    {
        return ids;
    }
    &[]
}

fn get_block(func: &llvm_ir::Function, id: llvm_ir::BlockId) -> Option<&llvm_ir::BasicBlock> {
    func.blocks.iter().find(|block| block.id == id)
}

fn get_inst_by_result(func: &llvm_ir::Function, id: llvm_ir::ValueId) -> Option<&llvm_ir::Inst> {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .find(|inst| inst.result == Some(id))
}

fn is_loop_preheader(func: &llvm_ir::Function, block: &llvm_ir::BasicBlock) -> bool {
    let llvm_ir::Terminator::Br { target: header, .. } = block.terminator else {
        return false;
    };
    get_block(func, header)
        .map(|header| is_loop_header(func, header))
        .unwrap_or(false)
}

fn is_loop_header(func: &llvm_ir::Function, block: &llvm_ir::BasicBlock) -> bool {
    let llvm_ir::Terminator::CondBr {
        true_target,
        false_target,
        ..
    } = block.terminator
    else {
        return false;
    };
    predecessor_count(func, block.id) > 1
        && (block_reaches(func, true_target, block.id, &mut HashSet::new())
            || block_reaches(func, false_target, block.id, &mut HashSet::new()))
}

fn shared_branch_join(
    func: &llvm_ir::Function,
    true_target: llvm_ir::BlockId,
    false_target: llvm_ir::BlockId,
) -> Option<llvm_ir::BlockId> {
    let true_block = get_block(func, true_target)?;
    let false_block = get_block(func, false_target)?;
    match (&true_block.terminator, &false_block.terminator) {
        (
            llvm_ir::Terminator::Br {
                target: true_join, ..
            },
            llvm_ir::Terminator::Br {
                target: false_join, ..
            },
        ) if true_join == false_join => Some(*true_join),
        _ => None,
    }
}

fn block_reaches(
    func: &llvm_ir::Function,
    start: llvm_ir::BlockId,
    goal: llvm_ir::BlockId,
    visiting: &mut HashSet<llvm_ir::BlockId>,
) -> bool {
    if start == goal {
        return true;
    }
    if !visiting.insert(start) {
        return false;
    }
    let reaches = match get_block(func, start).map(|block| &block.terminator) {
        Some(llvm_ir::Terminator::Br { target, .. }) => {
            block_reaches(func, *target, goal, visiting)
        }
        Some(llvm_ir::Terminator::CondBr {
            true_target,
            false_target,
            ..
        }) => {
            block_reaches(func, *true_target, goal, visiting)
                || block_reaches(func, *false_target, goal, visiting)
        }
        _ => false,
    };
    visiting.remove(&start);
    reaches
}

fn branch_condition_expr(
    func: &llvm_ir::Function,
    block: &llvm_ir::BasicBlock,
    cond: &llvm_ir::Operand,
    require_header_only: bool,
) -> (String, Option<llvm_ir::ValueId>) {
    let Some(cond_id) = operand_value_id(cond) else {
        return (format_operand_with_function(func, cond), None);
    };
    let Some(inst) = get_inst_by_result(func, cond_id) else {
        return (format_operand_with_function(func, cond), None);
    };
    if !matches!(inst.op, llvm_ir::InstOp::Cmp { .. }) {
        return (format_operand_with_function(func, cond), None);
    }
    if require_header_only && !header_only_contains_skipped_cmp(block, Some(cond_id)) {
        return (format_operand_with_function(func, cond), None);
    }
    match &inst.op {
        llvm_ir::InstOp::Cmp { pred, lhs, rhs } => (
            format!(
                "{} {} {}",
                format_operand_with_function(func, lhs),
                cmp_pred_symbol(*pred),
                format_operand_with_function(func, rhs)
            ),
            Some(cond_id),
        ),
        _ => (format_operand_with_function(func, cond), None),
    }
}

fn header_only_contains_skipped_cmp(
    block: &llvm_ir::BasicBlock,
    skip_result: Option<llvm_ir::ValueId>,
) -> bool {
    let Some(skip_result) = skip_result else {
        return false;
    };
    block
        .insts
        .iter()
        .all(|inst| inst.result == Some(skip_result))
}

fn operand_value_id(operand: &llvm_ir::Operand) -> Option<llvm_ir::ValueId> {
    match operand {
        llvm_ir::Operand::Value(id) => Some(*id),
        llvm_ir::Operand::Const(_) => None,
    }
}

fn format_operand_with_function(func: &llvm_ir::Function, operand: &llvm_ir::Operand) -> String {
    let value_names = build_value_names(func);
    format_operand(&value_names, operand)
}

fn predecessor_count(func: &llvm_ir::Function, target: llvm_ir::BlockId) -> usize {
    func.blocks
        .iter()
        .map(|block| match &block.terminator {
            llvm_ir::Terminator::Br { target: succ, .. } => usize::from(*succ == target),
            llvm_ir::Terminator::CondBr {
                true_target,
                false_target,
                ..
            } => usize::from(*true_target == target) + usize::from(*false_target == target),
            llvm_ir::Terminator::Ret { .. } | llvm_ir::Terminator::Switch { .. } => 0,
        })
        .sum()
}

fn build_value_type_map(func: &llvm_ir::Function) -> HashMap<llvm_ir::ValueId, llvm_ir::Type> {
    let mut types = HashMap::new();
    for param in &func.params {
        types.insert(param.id, param.ty.clone());
    }
    for block in &func.blocks {
        for param in &block.params {
            types.insert(param.id, param.ty.clone());
        }
        for inst in &block.insts {
            if let Some(id) = inst.result {
                types.insert(id, inst.ty.clone());
            }
        }
    }
    types
}

fn infer_tile_dims_from_scalar_store(
    value_id: llvm_ir::ValueId,
    inst_by_result: &HashMap<llvm_ir::ValueId, &llvm_ir::Inst>,
    value_types: &HashMap<llvm_ir::ValueId, llvm_ir::Type>,
) -> Option<(usize, usize)> {
    let load_inst = inst_by_result.get(&value_id)?;
    let llvm_ir::InstOp::Load {
        ptr: llvm_ir::Operand::Value(tile_ptr_id),
    } = &load_inst.op
    else {
        return None;
    };
    let tile_gep = inst_by_result.get(tile_ptr_id)?;
    let llvm_ir::InstOp::Gep {
        base: llvm_ir::Operand::Value(tile_id),
        ..
    } = &tile_gep.op
    else {
        return None;
    };
    let tile_ty = value_types.get(tile_id)?;
    tile_shape_from_type(tile_ty)
}

fn infer_tile_dims_from_direct_loop_store(
    func: &llvm_ir::Function,
    store_block: &llvm_ir::BasicBlock,
    inst_by_result: &HashMap<llvm_ir::ValueId, &llvm_ir::Inst>,
) -> Option<(usize, usize)> {
    // Find col_header: the block the store branches to
    let col_header_id = match &store_block.terminator {
        llvm_ir::Terminator::Br { target, .. } => *target,
        _ => return None,
    };

    let col_header = get_block(func, col_header_id)?;
    
    // Infer cols from col_header's loop bound (first param if exists)
    let cols = if col_header.params.is_empty() {
        return None;
    } else {
        // After phi_alias, col_header may have only 1 param (column loop)
        let col_loop_param = col_header.params.last()?.id;
        infer_loop_bound_from_header(col_header, col_loop_param, inst_by_result)?
    };

    // Find row_header via col_header's false branch -> row_latch -> row_header
    let llvm_ir::Terminator::CondBr {
        false_target: row_latch_id,
        ..
    } = &col_header.terminator
    else {
        return None;
    };
    let row_latch = get_block(func, *row_latch_id)?;
    let llvm_ir::Terminator::Br {
        target: row_header_id,
        ..
    } = &row_latch.terminator
    else {
        return None;
    };
    let row_header = get_block(func, *row_header_id)?;
    if row_header.params.is_empty() {
        return None;
    }
    let rows = infer_loop_bound_from_header(row_header, row_header.params[0].id, inst_by_result)?;

    if !block_reaches(func, col_header_id, store_block.id, &mut HashSet::new())
        || !block_reaches(func, *row_header_id, *row_latch_id, &mut HashSet::new())
    {
        return None;
    }

    Some((rows, cols))
}

fn infer_loop_bound_from_header(
    header: &llvm_ir::BasicBlock,
    loop_param: llvm_ir::ValueId,
    inst_by_result: &HashMap<llvm_ir::ValueId, &llvm_ir::Inst>,
) -> Option<usize> {
    let llvm_ir::Terminator::CondBr {
        cond: llvm_ir::Operand::Value(cond_id),
        ..
    } = &header.terminator
    else {
        return None;
    };
    let cmp_inst = inst_by_result.get(cond_id)?;
    let llvm_ir::InstOp::Cmp { pred, lhs, rhs } = &cmp_inst.op else {
        return None;
    };
    match (pred, lhs, rhs) {
        (
            llvm_ir::CmpPred::Slt,
            llvm_ir::Operand::Value(value_id),
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(bound)),
        ) if *value_id == loop_param && *bound >= 0 => Some(*bound as usize),
        (
            llvm_ir::CmpPred::Sle,
            llvm_ir::Operand::Value(value_id),
            llvm_ir::Operand::Const(llvm_ir::Constant::Int(bound)),
        ) if *value_id == loop_param && *bound >= 0 => Some((*bound as usize) + 1),
        _ => None,
    }
}

fn tile_shape_from_type(ty: &llvm_ir::Type) -> Option<(usize, usize)> {
    let llvm_ir::Type::Ptr {
        addr_space: llvm_ir::AddressSpace::Private,
        pointee,
    } = ty
    else {
        return None;
    };
    let dims = array_dims(pointee);
    match dims.as_slice() {
        [cols] => Some((1, *cols)),
        [rows, cols, ..] => Some((*rows, *cols)),
        _ => None,
    }
}
