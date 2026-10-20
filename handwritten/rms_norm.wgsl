// rms_norm: out[r][c] = x[r][c] * (1 / sqrt(mean(x[r]^2) + eps)) * w[c].
//
// One workgroup per row, the sum of squares a workgroup tree reduction. This is
// the same reduction shape as softmax with a different fold, which is what the
// `TileReduce` primitive has to cover.

struct Params {
    dims: vec4<u32>,
    scalars: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const WG: u32 = 256u;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    let cols = p.dims.y;
    let base = wid.x * cols;

    var sum = 0.0f;
    for (var c = t; c < cols; c += WG) {
        let v = x[base + c];
        sum += v * v;
    }
    red[t] = sum;
    workgroupBarrier();
    var stride = WG >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (t < stride) {
            red[t] = red[t] + red[t + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = red[0] / f32(cols);
    workgroupBarrier();
    let inv = 1.0f / sqrt(mean + p.scalars.y);

    for (var c = t; c < cols; c += WG) {
        out[base + c] = x[base + c] * inv * w[c];
    }
}
