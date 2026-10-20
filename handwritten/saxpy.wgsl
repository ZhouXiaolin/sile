// saxpy reference: out = a * x + y over a flat array.
//
// One thread per element. `dims.x` is the element count, `scalars.x` is `a`.
// The same program is `crates/sile/examples/saxpy.rs` flattened.

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
    let i = gid.x;
    if (i >= p.dims.x) {
        return;
    }
    out[i] = p.scalars.x * x[i] + y[i];
}
