use std::collections::HashMap;

use sile_core::{Error, Result};
use sile_llir as llir;
use sile_llir::Function as LlirFunction;

use crate::cpu::codegen::generate as generate_llir_c_source;
use crate::metal::codegen::generate as generate_llir_metal_source;
use crate::{BackendArtifact, CodegenTarget};

pub fn run(llir: &LlirFunction, target: CodegenTarget) -> Result<BackendArtifact> {
    let artifact = match target {
        CodegenTarget::C => BackendArtifact::CSource(generate_llir_c_source(llir)?),
        CodegenTarget::Metal => BackendArtifact::MetalSource(generate_llir_metal_source(llir)?),
    };
    Ok(artifact)
}

pub fn build_value_names(func: &llir::Function) -> HashMap<llir::ValueId, String> {
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

pub fn build_param_indices(func: &llir::Function) -> HashMap<llir::ValueId, usize> {
    func.params
        .iter()
        .enumerate()
        .map(|(idx, param)| (param.id, idx))
        .collect()
}

pub fn value_name(value_names: &HashMap<llir::ValueId, String>, id: llir::ValueId) -> String {
    value_names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("v{}", id.0))
}

pub fn format_operand(
    value_names: &HashMap<llir::ValueId, String>,
    operand: &llir::Operand,
) -> String {
    match operand {
        llir::Operand::Value(id) => value_name(value_names, *id),
        llir::Operand::Const(llir::Constant::Int(value)) => value.to_string(),
        llir::Operand::Const(llir::Constant::Float(value)) => float_literal(*value),
        llir::Operand::Const(llir::Constant::Bool(value)) => {
            if *value {
                "true".into()
            } else {
                "false".into()
            }
        }
    }
}

pub fn block_param_assignments(
    func: &llir::Function,
    value_names: &HashMap<llir::ValueId, String>,
    target: llir::BlockId,
    args: &[llir::Operand],
) -> Vec<(String, String)> {
    let Some(block) = func.blocks.iter().find(|block| block.id == target) else {
        return Vec::new();
    };
    block
        .params
        .iter()
        .zip(args.iter())
        .map(|(param, arg)| {
            (
                value_name(value_names, param.id),
                format_operand(value_names, arg),
            )
        })
        .collect()
}

pub fn array_dims(ty: &llir::Type) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut current = ty;
    while let llir::Type::Array { len, elem } = current {
        dims.push(*len);
        current = elem;
    }
    dims
}

pub fn bin_op_symbol(op: llir::BinOp) -> &'static str {
    match op {
        llir::BinOp::Add => "+",
        llir::BinOp::Sub => "-",
        llir::BinOp::Mul => "*",
        llir::BinOp::Div => "/",
        llir::BinOp::And => "&",
        llir::BinOp::Or => "|",
    }
}

pub fn cmp_pred_symbol(pred: llir::CmpPred) -> &'static str {
    match pred {
        llir::CmpPred::Eq => "==",
        llir::CmpPred::Ne => "!=",
        llir::CmpPred::Slt | llir::CmpPred::Olt => "<",
        llir::CmpPred::Sle | llir::CmpPred::Ole => "<=",
        llir::CmpPred::Sgt | llir::CmpPred::Ogt => ">",
        llir::CmpPred::Sge | llir::CmpPred::Oge => ">=",
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
    inst: &llir::Inst,
    value_name: FValue,
    format_operand: FOperand,
) -> Option<String>
where
    FValue: Fn(llir::ValueId) -> String,
    FOperand: Fn(&llir::Operand) -> String,
{
    match &inst.op {
        llir::InstOp::Bin { op, lhs, rhs } => inst.result.map(|id| {
            format!(
                "{} = {} {} {};",
                value_name(id),
                format_operand(lhs),
                bin_op_symbol(*op),
                format_operand(rhs)
            )
        }),
        llir::InstOp::Cmp { pred, lhs, rhs } => inst.result.map(|id| {
            format!(
                "{} = {} {} {};",
                value_name(id),
                format_operand(lhs),
                cmp_pred_symbol(*pred),
                format_operand(rhs)
            )
        }),
        llir::InstOp::Select {
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
        llir::InstOp::Load { ptr } => inst
            .result
            .map(|id| format!("{} = *({});", value_name(id), format_operand(ptr))),
        llir::InstOp::Store { ptr, value } => Some(format!(
            "*({}) = {};",
            format_operand(ptr),
            format_operand(value)
        )),
        llir::InstOp::Cast { value, .. } => inst
            .result
            .map(|id| format!("{} = {};", value_name(id), format_operand(value))),
        llir::InstOp::Gep { base, indices } => inst.result.map(|id| {
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

pub fn emit_value_decls<F>(func: &llir::Function, mut emit_decl: F) -> bool
where
    F: FnMut(llir::ValueId, &llir::Type),
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

pub fn infer_tile_plan(func: &llir::Function) -> Option<TilePlan> {
    for block in &func.blocks {
        for inst in &block.insts {
            if let llir::InstOp::Call { func: callee, args } = &inst.op {
                if callee != "tile_store_2d_f32" {
                    continue;
                }
                let [
                    llir::Operand::Value(buf_id),
                    _,
                    _,
                    _,
                    llir::Operand::Const(llir::Constant::Int(rows)),
                    llir::Operand::Const(llir::Constant::Int(cols)),
                    _,
                ] = args.as_slice()
                else {
                    continue;
                };
                let output_param = func.params.iter().position(|param| param.id == *buf_id)?;
                return Some(TilePlan {
                    output_param,
                    rows: *rows as usize,
                    cols: *cols as usize,
                });
            }
        }
    }

    let inst_by_result = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| inst.result.map(|id| (id, inst)))
        .collect::<HashMap<_, _>>();
    let value_types = build_value_type_map(func);

    for block in &func.blocks {
        for inst in &block.insts {
            let llir::InstOp::Store {
                ptr: llir::Operand::Value(ptr_id),
                value: llir::Operand::Value(value_id),
            } = &inst.op
            else {
                continue;
            };
            let Some(ptr_inst) = inst_by_result.get(ptr_id) else {
                continue;
            };
            let llir::InstOp::Gep {
                base: llir::Operand::Value(buf_id),
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
                continue;
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
    fn emit_block_insts(&mut self, block: &llir::BasicBlock) -> Result<()>;
    fn emit_block_param_assignments(&mut self, target: llir::BlockId, args: &[llir::Operand]);
    fn format_operand(&self, operand: &llir::Operand) -> String;
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
    func: &llir::Function,
    start: llir::BlockId,
    stop_targets: &[llir::BlockId],
    messages: StructuredCfgMessages,
) -> Result<Option<llir::BlockId>> {
    let mut current = start;
    loop {
        let block = get_block(func, current)
            .cloned()
            .ok_or_else(|| Error::Compile(format!("missing LLIR block {:?}", current)))?;

        if let llir::Terminator::Br { target, args } = &block.terminator
            && stop_targets.contains(target)
        {
            emitter.emit_block_insts(&block)?;
            emitter.emit_block_param_assignments(*target, args);
            return Ok(Some(*target));
        }

        if is_loop_preheader(func, &block) {
            current = emit_structured_loop_preheader(emitter, func, &block, messages)?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        if matches!(block.terminator, llir::Terminator::CondBr { .. }) {
            current = emit_structured_loop_header(emitter, func, &block, messages)?;
            if stop_targets.contains(&current) {
                return Ok(Some(current));
            }
            continue;
        }

        emitter.emit_block_insts(&block)?;
        match &block.terminator {
            llir::Terminator::Br { target, args } => {
                emitter.emit_block_param_assignments(*target, args);
                if stop_targets.contains(target) {
                    return Ok(Some(*target));
                }
                current = *target;
            }
            llir::Terminator::Ret { value: None } => {
                emitter.writeln("return;");
                return Ok(None);
            }
            llir::Terminator::Ret { value: Some(value) } => {
                emitter.writeln(&format!("return {};", emitter.format_operand(value)));
                return Ok(None);
            }
            llir::Terminator::CondBr { .. } => {
                return Err(Error::Compile(messages.unsupported_cond_br.into()));
            }
            llir::Terminator::Switch { .. } => {
                return Err(Error::Compile(messages.unsupported_switch.into()));
            }
        }
    }
}

fn emit_structured_loop_preheader<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llir::Function,
    preheader: &llir::BasicBlock,
    messages: StructuredCfgMessages,
) -> Result<llir::BlockId> {
    let llir::Terminator::Br {
        target: header_id,
        args: header_args,
    } = &preheader.terminator
    else {
        return Err(Error::Compile(messages.preheader_must_branch.into()));
    };
    let header = get_block(func, *header_id)
        .cloned()
        .ok_or_else(|| Error::Compile(messages.missing_loop_header.into()))?;
    let llir::Terminator::CondBr {
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
    emitter.emit_block_param_assignments(*header_id, header_args);
    emit_structured_loop_header_impl(
        emitter,
        func,
        &header,
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
    func: &llir::Function,
    header: &llir::BasicBlock,
    messages: StructuredCfgMessages,
) -> Result<llir::BlockId> {
    let llir::Terminator::CondBr {
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
        cond,
        *true_target,
        true_args,
        *false_target,
        false_args,
        messages,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_structured_loop_header_impl<E: StructuredEmitter>(
    emitter: &mut E,
    func: &llir::Function,
    header: &llir::BasicBlock,
    cond: &llir::Operand,
    true_target: llir::BlockId,
    true_args: &[llir::Operand],
    false_target: llir::BlockId,
    false_args: &[llir::Operand],
    messages: StructuredCfgMessages,
) -> Result<llir::BlockId> {
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
    let backedge = emit_structured_from(emitter, func, true_target, &[header.id], messages)?;
    if backedge != Some(header.id) {
        return Err(Error::Compile(messages.loop_backedge_mismatch.into()));
    }
    emitter.indent_dec();
    emitter.writeln("}");

    Ok(false_target)
}

fn get_block(func: &llir::Function, id: llir::BlockId) -> Option<&llir::BasicBlock> {
    func.blocks.iter().find(|block| block.id == id)
}

fn is_loop_preheader(func: &llir::Function, block: &llir::BasicBlock) -> bool {
    let llir::Terminator::Br { target: header, .. } = block.terminator else {
        return false;
    };
    matches!(
        get_block(func, header).map(|header| &header.terminator),
        Some(llir::Terminator::CondBr { .. })
    )
}

fn build_value_type_map(func: &llir::Function) -> HashMap<llir::ValueId, llir::Type> {
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
    value_id: llir::ValueId,
    inst_by_result: &HashMap<llir::ValueId, &llir::Inst>,
    value_types: &HashMap<llir::ValueId, llir::Type>,
) -> Option<(usize, usize)> {
    let load_inst = inst_by_result.get(&value_id)?;
    let llir::InstOp::Load {
        ptr: llir::Operand::Value(tile_ptr_id),
    } = &load_inst.op
    else {
        return None;
    };
    let tile_gep = inst_by_result.get(tile_ptr_id)?;
    let llir::InstOp::Gep {
        base: llir::Operand::Value(tile_id),
        ..
    } = &tile_gep.op
    else {
        return None;
    };
    let tile_ty = value_types.get(tile_id)?;
    tile_shape_from_type(tile_ty)
}

fn tile_shape_from_type(ty: &llir::Type) -> Option<(usize, usize)> {
    let llir::Type::Ptr {
        addr_space: llir::AddressSpace::Private,
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
