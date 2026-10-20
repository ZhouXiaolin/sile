// softmax over the last axis: one workgroup per row.
//
// `dims.x` is the row count, `dims.y` the column count. The row maximum and the
// row sum are workgroup tree reductions: the tile-level operation stays whole
// and the workgroup carries the parallelism, which is the shape M3 has to
// reproduce.

struct Params {
    dims: vec4<u32>,
    scalars: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const WG: u32 = 256u;

var<workgroup> red: array<f32, 256>;

// Fold `red` down to red[0]. Every thread returns with the barrier passed, so
// the caller may read red[0] and then reuse the array.
fn tree_reduce(lid: u32) -> f32 {
    var stride = WG >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid < stride) {
            red[lid] = red[lid] + red[lid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    return red[0];
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    let cols = p.dims.y;
    let base = wid.x * cols;

    var rowmax = -3.0e38f;
    for (var c = t; c < cols; c += WG) {
        rowmax = max(rowmax, x[base + c]);
    }
    red[t] = rowmax;
    workgroupBarrier();
    var stride = WG >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (t < stride) {
            red[t] = max(red[t], red[t + stride]);
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    rowmax = red[0];
    workgroupBarrier();

    var sum = 0.0f;
    for (var c = t; c < cols; c += WG) {
        sum += exp(x[base + c] - rowmax);
    }
    red[t] = sum;
    workgroupBarrier();
    let total = tree_reduce(t);
    let inv = 1.0f / total;

    for (var c = t; c < cols; c += WG) {
        out[base + c] = exp(x[base + c] - rowmax) * inv;
    }
}
