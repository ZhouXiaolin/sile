// gemm: z[m][n] = sum_k x[m][k] * y[k][n].
//
// One workgroup per 32x16 output tile, 8x8 threads, each thread owning a 4x4
// sub-block held in registers. The K axis is walked in 16-deep steps through
// two `var<workgroup>` panels, so one barrier per step. `dims` is (m, n, k).

struct Params {
    dims: vec4<u32>,
    scalars: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const TM: u32 = 32u;   // tile rows
const TN: u32 = 16u;   // tile cols
const BK: u32 = 16u;   // contraction step
const TX: u32 = 4u;    // threads along cols
const TY: u32 = 8u;    // threads along cols
const SUB: u32 = 4u;   // elements per thread per axis

var<workgroup> xs: array<f32, 32 * 16>;
var<workgroup> ys: array<f32, 16 * 16>;

@compute @workgroup_size(4, 8)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = p.dims.x;
    let n = p.dims.y;
    let k = p.dims.z;

    let tile_row = wid.y * TM;
    let tile_col = wid.x * TN;
    let tx = lid.x;
    let ty = lid.y;

    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < SUB; i += 1u) {
        for (var j = 0u; j < SUB; j += 1u) {
            acc[i][j] = 0.0f;
        }
    }

    // One step loads TM*BK x-panel and BK*TN y-panel with TX*TY threads.
    let threads = TX * TY;
    let x_elems = TM * BK;
    let y_elems = BK * TN;
    let tid = ty * TX + tx;

    for (var k0 = 0u; k0 < k; k0 += BK) {
        for (var e = tid; e < x_elems; e += threads) {
            let r = e / BK;
            let c = e % BK;
            let gr = tile_row + r;
            let gc = k0 + c;
            xs[e] = select(0.0f, x[gr * k + gc], gr < m && gc < k);
        }
        for (var e = tid; e < y_elems; e += threads) {
            let r = e / TN;
            let c = e % TN;
            let gr = k0 + r;
            let gc = tile_col + c;
            ys[e] = select(0.0f, y[gr * n + gc], gr < k && gc < n);
        }
        workgroupBarrier();

        for (var p1 = 0u; p1 < BK; p1 += 1u) {
            var xv: array<f32, 4>;
            var yv: array<f32, 4>;
            for (var i = 0u; i < SUB; i += 1u) {
                xv[i] = xs[(ty * SUB + i) * BK + p1];
            }
            for (var j = 0u; j < SUB; j += 1u) {
                yv[j] = ys[p1 * TN + tx * SUB + j];
            }
            for (var i = 0u; i < SUB; i += 1u) {
                for (var j = 0u; j < SUB; j += 1u) {
                    acc[i][j] += xv[i] * yv[j];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < SUB; i += 1u) {
        let gr = tile_row + ty * SUB + i;
        if (gr >= m) {
            continue;
        }
        for (var j = 0u; j < SUB; j += 1u) {
            let gc = tile_col + tx * SUB + j;
            if (gc < n) {
                out[gr * n + gc] = acc[i][j];
            }
        }
    }
}
