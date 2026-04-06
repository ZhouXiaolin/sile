use crate::{MirFunction, MirOp, MirType};

pub fn run(mut func: MirFunction) -> MirFunction {
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            match &mut inst.op {
                MirOp::TileReduce { axis, .. } => {
                    *axis = axis.rem_euclid(2);
                }
                MirOp::TileLoad {
                    buf,
                    stride_shape_idx,
                    ..
                }
                | MirOp::TileStore {
                    buf,
                    stride_shape_idx,
                    ..
                } => {
                    let rank = match func.types.get(buf) {
                        Some(MirType::Buffer { rank }) => *rank,
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
