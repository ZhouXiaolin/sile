//! AIR-behaviour probe for the flash-attention Metal bug: four spellings of
//! the same p @ v accumulation, to bisect which construct the compiler
//! miscompiles (raw binding / emitter text / i32 loops / plain stack arrays).

use sile_backend::metal::MetalBackend;
use sile_core::{KernelArg, LaunchConfig};
use sile_lsir::ir::Function;

#[path = "common/mod.rs"]
mod common;

fn main() -> sile::Result<()> {
    let backend = MetalBackend::new()?;
    let func = Function {
        name: "probe".into(),
        params: vec![],
        blocks: vec![],
        entry: sile_lsir::ir::BlockId(0),
        metadata: vec![],
    };

    let source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sile_kernel_probe(
    device float* vin [[buffer(0)]],
    device float* out [[buffer(1)]])
{
  // A: raw copy — binding sanity.
  for (int i = 0; i < 32; ++i) out[i] = vin[i];

  // B: the emitter's exact spelling: thread arrays via pointer alias, i64 loops.
  thread float tb_storage[32]; thread float (*tb) = tb_storage;
  thread float tp_storage[16]; thread float (*tp) = tp_storage;
  thread float ta_storage[32]; thread float (*ta) = ta_storage;
  for (int64_t iv = 0; iv < 32; ++iv) { tb[iv] = vin[iv]; }
  for (int64_t iv = 0; iv < 16; ++iv) { tp[iv] = 1.0f; }
  for (int64_t mi = 0; mi < 32; ++mi) { ta[mi] = 0.0f; }
  for (int64_t mo = 0; mo < 4; ++mo) {
    for (int64_t no = 0; no < 8; ++no) {
      float racc = ta[mo * 8 + no];
      for (int64_t ko = 0; ko < 4; ++ko) {
        racc += tp[mo * 4 + ko] * tb[ko * 8 + no];
      }
      ta[mo * 8 + no] = racc;
    }
  }
  for (int64_t iv = 0; iv < 32; ++iv) { out[32 + iv] = ta[iv]; }

  // C: same, plain stack arrays and i32 loops.
  float cb[32]; float cp[16]; float ca[32] = {0};
  for (int i = 0; i < 32; ++i) cb[i] = vin[i];
  for (int i = 0; i < 16; ++i) cp[i] = 1.0f;
  for (int mo = 0; mo < 4; ++mo)
    for (int no = 0; no < 8; ++no) {
      float racc = ca[mo * 8 + no];
      for (int ko = 0; ko < 4; ++ko) racc += cp[mo * 4 + ko] * cb[ko * 8 + no];
      ca[mo * 8 + no] = racc;
    }
  for (int i = 0; i < 32; ++i) out[64 + i] = ca[i];

  // D: the real kernel's store pattern: divide by a per-row carry indexed i/8.
  thread float cl_storage[4]; thread float (*cl) = cl_storage;
  for (int64_t iv = 0; iv < 4; ++iv) { cl[iv] = 4.0f; }
  for (int64_t iv = 0; iv < 32; ++iv) {
    int64_t row = iv / 8;
    out[96 + iv] = ta[iv] / cl[row];
  }

  // E: thread-pointer arrays + i32 loops — isolates the pointer aliasing.
  thread float eb_storage[32]; thread float (*eb) = eb_storage;
  thread float ep_storage[16]; thread float (*ep) = ep_storage;
  thread float ea_storage[32]; thread float (*ea) = ea_storage;
  for (int i = 0; i < 32; ++i) { eb[i] = vin[i]; }
  for (int i = 0; i < 16; ++i) { ep[i] = 1.0f; }
  for (int mi = 0; mi < 32; ++mi) { ea[mi] = 0.0f; }
  for (int mo = 0; mo < 4; ++mo)
    for (int no = 0; no < 8; ++no) {
      float racc = ea[mo * 8 + no];
      for (int ko = 0; ko < 4; ++ko) racc += ep[mo * 4 + ko] * eb[ko * 8 + no];
      ea[mo * 8 + no] = racc;
    }
  for (int i = 0; i < 32; ++i) out[128 + i] = ea[i];

  // F: plain stack arrays + i64 loops — isolates the index width.
  float fb[32]; float fp[16]; float fa[32] = {0};
  for (int64_t i = 0; i < 32; ++i) { fb[i] = vin[i]; }
  for (int64_t i = 0; i < 16; ++i) { fp[i] = 1.0f; }
  for (int64_t mo = 0; mo < 4; ++mo)
    for (int64_t no = 0; no < 8; ++no) {
      float racc = fa[mo * 8 + no];
      for (int64_t ko = 0; ko < 4; ++ko) racc += fp[mo * 4 + ko] * fb[ko * 8 + no];
      fa[mo * 8 + no] = racc;
    }
  for (int64_t i = 0; i < 32; ++i) out[160 + i] = fa[i];

  // G: single-variable i64 subscripts on a thread array (the iv-driven
  // TileLoad/TileStore shape the structured emitter produces).
  thread float gb_storage[32]; thread float (*gb) = gb_storage;
  for (int64_t iv = 0; iv < 32; ++iv) { gb[iv] = vin[iv]; }
  for (int64_t iv = 0; iv < 32; ++iv) { out[192 + iv] = gb[iv]; }

  // I: int64_t-element arrays with i64 subscripts (the iota/offs tile shape).
  thread int64_t ib_storage[32]; thread int64_t (*ib) = ib_storage;
  thread int64_t im_storage[1024]; thread int64_t (*im) = im_storage;
  for (int i = 0; i < 32; ++i) { ib[i] = i; }
  for (int64_t iv = 0; iv < 1024; ++iv) { im[iv] = ib[iv % 32]; }
  int64_t acc_i = 0;
  for (int64_t iv = 0; iv < 1024; ++iv) { acc_i += im[iv] * (iv % 7); }
  out[256] = (float)(acc_i % 1000);

  // J: the causal-mask shape, i64: iota arrays, broadcast, select inside if.
  thread int64_t jb_storage[32]; thread int64_t (*jb) = jb_storage;
  thread int64_t jn_storage[1024]; thread int64_t (*jn) = jn_storage;
  thread int64_t jm_storage[1024]; thread int64_t (*jm) = jm_storage;
  for (int i = 0; i < 32; ++i) { jb[i] = i; }
  for (int64_t iv = 0; iv < 1024; ++iv) { jn[iv] = jb[iv % 32]; }
  for (int64_t iv = 0; iv < 1024; ++iv) { jm[iv] = 5 + (int)(iv / 32); }
  int64_t true_i = 0;
  for (int64_t j = 0; j < 1; ++j) {
    if (j >= 0) {
      for (int64_t iv = 0; iv < 1024; ++iv) {
        int64_t offs_n = j * 32 + jn[iv];
        bool m = jm[iv] >= offs_n;
        true_i += m ? 1 : 0;
      }
    }
  }
  out[260] = (float)true_i;

  // K: the same mask shape fully spelled in int.
  int kb[32]; int kn[1024]; int km[1024];
  for (int i = 0; i < 32; ++i) kb[i] = i;
  for (int iv = 0; iv < 1024; ++iv) kn[iv] = kb[iv % 32];
  for (int iv = 0; iv < 1024; ++iv) km[iv] = 5 + iv / 32;
  int true_k = 0;
  for (int j = 0; j < 1; ++j) {
    if (j >= 0) {
      for (int iv = 0; iv < 1024; ++iv) {
        int offs_n = j * 32 + kn[iv];
        bool m = km[iv] >= offs_n;
        true_k += m ? 1 : 0;
      }
    }
  }
  out[264] = (float)true_k;

  // L: constant -inf select arm + add: the fast-math folding suspect.
  float v0 = vin[0];
  float sl = (vin[1] < 0.0f) ? 0.0f : ((-INFINITY));
  out[268] = v0 + sl;

  // M: the same -inf laundered through a thread array cell.
  thread float ns_storage[1]; thread float (*nsp) = ns_storage;
  nsp[0] = (-INFINITY);
  float sm = (vin[1] < 0.0f) ? 0.0f : nsp[0];
  out[272] = v0 + sm;

  // N1: bitcast constant.
  float n1 = (vin[1] < 0.0f) ? 0.0f : as_type<float>(0xff800000u);
  out[276] = v0 + n1;
  // N2: volatile launder.
  volatile float n2v = (-INFINITY);
  float n2 = (vin[1] < 0.0f) ? 0.0f : (float)n2v;
  out[280] = v0 + n2;
  // N3: metal::select builtin.
  float n3 = metal::select(0.0f, (-INFINITY), (bool)(vin[1] >= 0.0f));
  out[284] = v0 + n3;
  // N5: subtract +inf.
  float n5 = (vin[1] < 0.0f) ? 0.0f : (-INFINITY);
  out[288] = v0 - n5 * 0.0f + n5;  // keep it simple: v0 + n5 spelled as v0 - (-n5)? just test v0 + n5 again via separate var
  out[292] = v0 + (0.0f - n5);

  // O: the rewritten mask shape — select(mask, x, -inf), no add. The -inf
  // spells as a bitcast so the source stays on the fast-math compile path.
  float x0 = vin[0];
  float o = (vin[1] < 0.0f) ? x0 : as_type<float>(0xff800000u);
  out[296] = o;

  // P: the rewritten shape feeding the softmax consumers.
  float mx = metal::max(o, 0.0f);
  float q = (o == o) ? metal::exp2(o - mx) : 0.0f;
  out[300] = mx;
  out[304] = q;

  // H: i64 subscripts with an inner i64 no loop — the serial TileMma
  // operand shape, reading a plain stack array.
  float hb[32];
  for (int64_t i = 0; i < 32; ++i) hb[i] = vin[i];
  for (int64_t no = 0; no < 8; ++no) {
    float racc = 0.0f;
    for (int64_t ko = 0; ko < 4; ++ko) racc += hb[ko * 8 + no];
    out[224 + no] = racc;
  }
}
"#;

    let v: Vec<f32> = (0..32).map(|i| ((i as f32 + 2.0) * 0.37).sin() * 0.5).collect();
    let mut out = vec![0f32; 308];
    let device = sile::Device::metal();

    let v_arg = KernelArg {
        ptr: v.as_ptr() as *const u8,
        mut_ptr: v.as_ptr() as *mut u8,
        elem_size: 4,
        shape: &[32],
        device: &device,
        written: false,
    };
    let out_arg = KernelArg {
        ptr: out.as_ptr() as *const u8,
        mut_ptr: out.as_mut_ptr() as *mut u8,
        elem_size: 4,
        shape: &[308],
        device: &device,
        written: true,
    };

    backend.execute_source(
        &func,
        source,
        &[false, true],
        &[v_arg, out_arg],
        &[],
        &[],
        &LaunchConfig { grid: [1, 1, 1] },
        1,
    )?;

    for (name, base) in [("A raw", 0), ("B emitter", 32), ("C plain", 64), ("D divide", 96), ("E ptr+i32", 128), ("F plain+i64", 160), ("G iv-load", 192), ("H i64-expr", 224), ("I i64-elem", 256), ("J mask-i64", 260), ("K mask-i32", 264), ("L const-inf", 268), ("M array-inf", 272), ("N1 bitcast", 276), ("N2 volatile", 280), ("N3 mtl-sel", 284), ("N5 sub-inf", 288), ("N6 0-neg", 292), ("O sel-noadd", 296), ("P consumers", 300)] {
        println!("{name}: {:+.4}", out[base]);
    }
    println!("want col-sum/4 (D): {}", {
        let mut s = 0f32;
        for k in 0..4 {
            s += v[k * 8];
        }
        s / 4.0
    });
    Ok(())
}
