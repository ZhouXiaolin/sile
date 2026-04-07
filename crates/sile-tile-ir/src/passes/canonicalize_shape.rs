use crate::{TileIrFunction, TileIrOp, TileIrType};

pub fn run(mut func: TileIrFunction) -> TileIrFunction {
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            match &mut inst.op {
                TileIrOp::ReduceSum { axis, .. } | TileIrOp::ReduceMax { axis, .. } => {
                    *axis = axis.rem_euclid(2);
                }
                TileIrOp::LoadPtrTko {
                    buf,
                    stride_shape_idx,
                    ..
                }
                | TileIrOp::StorePtrTko {
                    buf,
                    stride_shape_idx,
                    ..
                } => {
                    let rank = match func.types.get(buf) {
                        Some(TileIrType::Buffer { rank }) => *rank,
                        _ => 1,
                    };
                    if rank <= 1 {
                        *stride_shape_idx = 0;
                    }
                }
                _ => {}
            }
        }
    }
    func
}
