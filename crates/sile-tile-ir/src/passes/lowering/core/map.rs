use sile_llvm_ir as llvm_ir;

use crate::ir::*;

pub(crate) fn llvm_ir_value(value: ValueId) -> llvm_ir::ValueId {
    llvm_ir::ValueId(value.0)
}

pub(crate) fn llvm_ir_block(block: BlockId) -> llvm_ir::BlockId {
    llvm_ir::BlockId(block.0)
}

pub(crate) fn llvm_ir_type(ty: &TileIrType) -> llvm_ir::Type {
    match ty {
        TileIrType::I64 => llvm_ir::Type::I64,
        TileIrType::F32 => llvm_ir::Type::F32,
        TileIrType::Tile { rows, cols } => tile_ptr_type(*rows, *cols),
        TileIrType::Buffer { .. } => {
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32)
        }
        TileIrType::Void => llvm_ir::Type::Void,
    }
}

pub(crate) fn tile_storage_type(rows: i64, cols: i64) -> llvm_ir::Type {
    llvm_ir::Type::array(
        rows as usize,
        llvm_ir::Type::array(cols as usize, llvm_ir::Type::F32),
    )
}

pub(crate) fn tile_ptr_type(rows: i64, cols: i64) -> llvm_ir::Type {
    llvm_ir::Type::ptr(
        llvm_ir::AddressSpace::Private,
        tile_storage_type(rows, cols),
    )
}

pub(crate) fn lower_bin_op(op: BinOp) -> llvm_ir::BinOp {
    match op {
        BinOp::Add => llvm_ir::BinOp::Add,
        BinOp::Sub => llvm_ir::BinOp::Sub,
        BinOp::Mul => llvm_ir::BinOp::Mul,
        BinOp::Div => llvm_ir::BinOp::Div,
    }
}

pub(crate) fn lower_cmp_pred(op: CmpOp) -> llvm_ir::CmpPred {
    match op {
        CmpOp::Lt => llvm_ir::CmpPred::Slt,
        CmpOp::Le => llvm_ir::CmpPred::Sle,
        CmpOp::Gt => llvm_ir::CmpPred::Sgt,
        CmpOp::Ge => llvm_ir::CmpPred::Sge,
        CmpOp::Eq => llvm_ir::CmpPred::Eq,
        CmpOp::Ne => llvm_ir::CmpPred::Ne,
    }
}

pub(crate) fn buffer_rank_of(value: ValueId, tile_ir: &TileIrFunction) -> usize {
    match tile_ir.types.get(&value) {
        Some(TileIrType::Buffer { rank }) => *rank,
        _ => 1,
    }
}

pub(crate) fn tile_dims_of(value: ValueId, tile_ir: &TileIrFunction) -> Option<(i64, i64)> {
    match tile_ir.types.get(&value) {
        Some(TileIrType::Tile { rows, cols }) => Some((*rows, *cols)),
        _ => None,
    }
}
