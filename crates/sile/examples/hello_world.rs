//! `hello_world`, ported from cutile-examples/examples/hello_world.rs.
//!
//! The kernel takes no tensor and computes nothing: it prints the program's id
//! and the grid extent. What it exercises is the print op, whose format string
//! and operand types are both part of the Tile IR.
//!
//! The original asks for `print_ir = true` on the entry; here the same text is
//! what `SILE_DUMP_DIR` writes, and there is no output to check against numpy,
//! because a print has no value.

use sile::Device;

#[sile::module]
mod kernels {
    use sile::ct::*;
    use sile::cuda_tile_print;

    #[sile::entry]
    pub fn hello_world_kernel() {
        let pid0 = program_id(0);
        let pid1 = program_id(1);
        let pid2 = program_id(2);
        let n0 = num_programs(0);
        let n1 = num_programs(1);
        let n2 = num_programs(2);
        cuda_tile_print!(
            "Hello, I am program <{}, {}, {}> in a kernel with <{}, {}, {}> programs.\n",
            pid0,
            pid1,
            pid2,
            n0,
            n1,
            n2
        );
    }
}

fn main() -> sile::Result<()> {
    let device = Device::default()?;
    let stream = device.create_stream()?;

    // A kernel with no tensor has no extent to derive a grid from, so the
    // launch states one: a single program, enough for the print to run once.
    kernels::hello_world_kernel()
        .grid((1, 1, 1))
        .apply(&stream)?;
    println!("hello_world ok on {device:?}");
    Ok(())
}
