// saxpy reference: out = a * x + y over a flat array.
//
// One thread per element. `dims.x` is the element count, `dims.y` is the
// per-row element stride of the launch grid — a 16M-element dispatch does not
// fit one grid dimension's 65535-workgroup cap, so the launch is rows of
// workgroups and the row index folds into the element index here.
// `scalars.x` is `a`. The same program is `crates/sile/examples/saxpy.rs`
// flattened.

struct Params {
    dims: vec4<u32>,
    scalars: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.dims.y + gid.x;
    if (i >= p.dims.x) {
        return;
    }
    out[i] = p.scalars.x * x[i] + y[i];
}
