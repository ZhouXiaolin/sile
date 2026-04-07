use sile_llvm_ir as llvm_ir;

use crate::ValueId;

use crate::passes::lowering::core::block::BlockLowerer;
use crate::passes::lowering::core::map::{llvm_ir_value, tile_ptr_type, tile_storage_type};

pub(crate) fn alloc_tile_result(
    builder: &mut BlockLowerer<'_>,
    result: ValueId,
    rows: i64,
    cols: i64,
) -> llvm_ir::Operand {
    let llir_id = llvm_ir_value(result);
    let name = format!("v{}", result.0);
    builder.with_current_insts(|ctx, _, out| {
        ctx.names.insert(llir_id, name.clone());
        ctx.operands
            .insert(result, llvm_ir::Operand::Value(llir_id));
        out.push(llvm_ir::Inst {
            result: Some(llir_id),
            result_name: Some(name),
            ty: tile_ptr_type(rows, cols),
            op: llvm_ir::InstOp::Alloca {
                alloc_ty: tile_storage_type(rows, cols),
                addr_space: llvm_ir::AddressSpace::Private,
            },
            metadata: vec![llvm_ir::Metadata::Alignment(16)],
        });
    });
    llvm_ir::Operand::Value(llir_id)
}
