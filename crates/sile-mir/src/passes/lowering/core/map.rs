use sile_llir as llir;

use crate::ir::*;

pub(crate) fn llir_value(value: ValueId) -> llir::ValueId {
    llir::ValueId(value.0)
}

pub(crate) fn llir_block(block: BlockId) -> llir::BlockId {
    llir::BlockId(block.0)
}

pub(crate) fn llir_type(ty: &MirType) -> llir::Type {
    match ty {
        MirType::I64 => llir::Type::I64,
        MirType::F32 => llir::Type::F32,
        MirType::Tile { rows, cols } => tile_ptr_type(*rows, *cols),
        MirType::Buffer { .. } => llir::Type::ptr(llir::AddressSpace::Global, llir::Type::F32),
        MirType::Void => llir::Type::Void,
    }
}

pub(crate) fn tile_storage_type(rows: i64, cols: i64) -> llir::Type {
    llir::Type::array(
        rows as usize,
        llir::Type::array(cols as usize, llir::Type::F32),
    )
}

pub(crate) fn tile_ptr_type(rows: i64, cols: i64) -> llir::Type {
    llir::Type::ptr(llir::AddressSpace::Private, tile_storage_type(rows, cols))
}

pub(crate) fn lower_bin_op(op: BinOp) -> llir::BinOp {
    match op {
        BinOp::Add => llir::BinOp::Add,
        BinOp::Sub => llir::BinOp::Sub,
        BinOp::Mul => llir::BinOp::Mul,
        BinOp::Div => llir::BinOp::Div,
    }
}

pub(crate) fn lower_cmp_pred(op: CmpOp) -> llir::CmpPred {
    match op {
        CmpOp::Lt => llir::CmpPred::Slt,
        CmpOp::Le => llir::CmpPred::Sle,
        CmpOp::Gt => llir::CmpPred::Sgt,
        CmpOp::Ge => llir::CmpPred::Sge,
        CmpOp::Eq => llir::CmpPred::Eq,
        CmpOp::Ne => llir::CmpPred::Ne,
    }
}

pub(crate) fn buffer_rank_of(value: ValueId, mir: &MirFunction) -> usize {
    match mir.types.get(&value) {
        Some(MirType::Buffer { rank }) => *rank,
        _ => 1,
    }
}

pub(crate) fn tile_dims_of(value: ValueId, mir: &MirFunction) -> Option<(i64, i64)> {
    match mir.types.get(&value) {
        Some(MirType::Tile { rows, cols }) => Some((*rows, *cols)),
        _ => None,
    }
}
