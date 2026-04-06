use sile_llir as llir;

use crate::ValueId;

use crate::lower_llir_core::block::BlockLowerer;
use crate::lower_llir_core::map::{llir_value, tile_ptr_type, tile_storage_type};

pub(crate) fn alloc_tile_result(
    builder: &mut BlockLowerer<'_>,
    result: ValueId,
    rows: i64,
    cols: i64,
) -> llir::Operand {
    let llir_id = llir_value(result);
    let name = format!("v{}", result.0);
    builder.with_current_insts(|ctx, _, out| {
        ctx.names.insert(llir_id, name.clone());
        ctx.operands.insert(result, llir::Operand::Value(llir_id));
        out.push(llir::Inst {
            result: Some(llir_id),
            result_name: Some(name),
            ty: tile_ptr_type(rows, cols),
            op: llir::InstOp::Alloca {
                alloc_ty: tile_storage_type(rows, cols),
                addr_space: llir::AddressSpace::Private,
            },
            metadata: vec![llir::Metadata::Alignment(16)],
        });
    });
    llir::Operand::Value(llir_id)
}
