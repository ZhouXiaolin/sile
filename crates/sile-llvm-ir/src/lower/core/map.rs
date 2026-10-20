use crate as llvm_ir;

use sile_tile_ir::function_ir::*;

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
        TileIrType::ShapeDesc { .. } => {
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Constant, llvm_ir::Type::I64)
        }
        TileIrType::Tile { rows, cols } => tile_ptr_type(*rows, *cols),
        TileIrType::Buffer { elem, .. } => {
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, buffer_elem_type(*elem))
        }
        // The general forms carry no storage of their own in a kernel this
        // lowering still emits; only their result types reach a target.
        TileIrType::Scalar(s) => llvm_ir_type(&scalar_to_tile_ir(*s)),
        TileIrType::Pointer(_) => {
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::I64)
        }
        TileIrType::TileOf(t) => match t.shape.as_slice() {
            [rows, cols] => tile_ptr_type(*rows, *cols),
            [rows] => tile_ptr_type(*rows, 1),
            _ => llvm_ir::Type::ptr(llvm_ir::AddressSpace::Private, llvm_ir::Type::F32),
        },
        // A view's storage is the tensor's: a pointer to its element type in
        // global memory, exactly as a bare buffer pointer was.
        TileIrType::TensorView(_)
        | TileIrType::PartitionView(_)
        | TileIrType::GatherScatterView(_)
        | TileIrType::StridedView(_) => {
            match tensor_view_of(ty).map(|tv| tv.element_type) {
                Some(ScalarType::F16) => {
                    llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F16)
                }
                _ => llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::F32),
            }
        }
        TileIrType::Token | TileIrType::Func(_) => {
            llvm_ir::Type::ptr(llvm_ir::AddressSpace::Global, llvm_ir::Type::I64)
        }
        TileIrType::WithPointerAttribute(base, _) => llvm_ir_type(base),
        TileIrType::Void => llvm_ir::Type::Void,
    }
}

/// The two scalar forms that carry storage of their own.
fn scalar_to_tile_ir(s: ScalarType) -> TileIrType {
    match s {
        ScalarType::I64 | ScalarType::I32 | ScalarType::I16 | ScalarType::I8 | ScalarType::I1 => {
            TileIrType::I64
        }
        _ => TileIrType::F32,
    }
}

/// The LLVM IR type of one element of a buffer's storage.
fn buffer_elem_type(elem: BufferElem) -> llvm_ir::Type {
    match elem {
        BufferElem::F32 => llvm_ir::Type::F32,
        BufferElem::F16 => llvm_ir::Type::F16,
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
        BinOp::Rem => llvm_ir::BinOp::Rem,
        BinOp::And => llvm_ir::BinOp::And,
        BinOp::Or => llvm_ir::BinOp::Or,
        BinOp::Xor => llvm_ir::BinOp::Xor,
        BinOp::Shl => llvm_ir::BinOp::Shl,
        BinOp::Shr => llvm_ir::BinOp::Shr,
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

/// The predicate a float comparison uses.
///
/// LLVM has one set of comparison instructions for integers and another for
/// floats, and `O` marks the ordered float one: a comparison with a NaN operand
/// is false, which is what makes a NaN mask a cleared element.
pub(crate) fn lower_float_cmp_pred(op: CmpOp) -> llvm_ir::CmpPred {
    match op {
        CmpOp::Lt => llvm_ir::CmpPred::Olt,
        CmpOp::Le => llvm_ir::CmpPred::Ole,
        CmpOp::Gt => llvm_ir::CmpPred::Ogt,
        CmpOp::Ge => llvm_ir::CmpPred::Oge,
        CmpOp::Eq => llvm_ir::CmpPred::Eq,
        CmpOp::Ne => llvm_ir::CmpPred::Ne,
    }
}

/// The predicate a comparison of these operands uses.
///
/// A comparison of two counts is a signed integer comparison and a comparison
/// of two elements is an ordered float one. The operands say which, so a kernel
/// writes `min(mask_start, k_seqlen_tiles)` and `max_tile(m_i, qk_max)` with the
/// same operator and each reaches the instruction it needs.
pub(crate) fn comparison_pred(
    op: CmpOp,
    lhs: ValueId,
    rhs: ValueId,
    tile_ir: &TileIrFunction,
) -> llvm_ir::CmpPred {
    let float = matches!(tile_ir.types.get(&lhs), Some(TileIrType::F32))
        || matches!(tile_ir.types.get(&rhs), Some(TileIrType::F32));
    if float {
        lower_float_cmp_pred(op)
    } else {
        lower_cmp_pred(op)
    }
}

pub(crate) fn buffer_rank_of(value: ValueId, tile_ir: &TileIrFunction) -> usize {
    match tile_ir.types.get(&value) {
        Some(TileIrType::Buffer { rank, .. }) => *rank,
        // A kernel parameter may also arrive as the tensor view CuTile names.
        // Its rank is the length of the shape the view describes.
        Some(TileIrType::TensorView(tv)) => tv.shape.len(),
        Some(TileIrType::PartitionView(pv)) => pv.tensor_view.shape.len(),
        Some(TileIrType::GatherScatterView(gv)) => gv.tensor_view.shape.len(),
        Some(TileIrType::StridedView(sv)) => sv.tensor_view.shape.len(),
        _ => 1,
    }
}

/// The tensor view behind a view value, peeled from whichever view kind it is.
pub(crate) fn tensor_view_of(ty: &TileIrType) -> Option<&TensorViewType> {
    match ty {
        TileIrType::TensorView(tv) => Some(tv),
        TileIrType::PartitionView(pv) => Some(&pv.tensor_view),
        TileIrType::GatherScatterView(gv) => Some(&gv.tensor_view),
        TileIrType::StridedView(sv) => Some(&sv.tensor_view),
        _ => None,
    }
}

/// The storage element type of a buffer value.
///
/// A load reads this and a store writes it; the tiles on either side stay f32,
/// so the conversion between the two is the buffer edge and lives where the
/// load and the store are emitted.
pub(crate) fn buffer_elem_of(value: ValueId, tile_ir: &TileIrFunction) -> llvm_ir::Type {
    match tile_ir.types.get(&value) {
        Some(TileIrType::Buffer { elem, .. }) => buffer_elem_type(*elem),
        // A view states its element type on the tensor it describes; reading
        // f32 there would re-type a half buffer's loads and stores.
        Some(ty) => match tensor_view_of(ty).map(|tv| tv.element_type) {
            Some(ScalarType::F16) => llvm_ir::Type::F16,
            _ => llvm_ir::Type::F32,
        },
        None => llvm_ir::Type::F32,
    }
}

pub(crate) fn tile_dims_of(value: ValueId, tile_ir: &TileIrFunction) -> Option<(i64, i64)> {
    match tile_ir.types.get(&value) {
        Some(TileIrType::Tile { rows, cols }) => Some((*rows, *cols)),
        _ => None,
    }
}
