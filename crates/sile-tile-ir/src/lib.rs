pub mod cutile;
pub mod ir;
pub mod opcode;

// The one Tile IR: the arena-structured port of cutile-ir. A kernel is lowered
// into a `Module` of it by `cutile::emit`, and every backend reads that module
// directly — there is no second, function-shaped Tile IR to keep in step.
pub use ir::*;
