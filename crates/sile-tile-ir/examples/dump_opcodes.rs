//! Print every CUDA Tile opcode name, one per line.
//!
//! The list is compared against `cutile-ir`'s table: the alignment is only
//! real if the two name sets are equal, so this makes that check a diff.

fn main() {
    for op in sile_tile_ir::function_ir::Opcode::ALL {
        println!("{}", op.name());
    }
}
