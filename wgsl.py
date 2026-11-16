"""Compile a trained Demosaic into WGSL compute shaders plus a dispatch list,
for the Rust executor in wgpu-run/.

The network is specialised for one fixed window size: every shape, weight
offset and flag is a compile-time constant in the generated shader, which is
what lets the GPU compiler unroll loops and fold index math. Images of any size
are run as overlapping windows (see wgpu-run).

    write(out_dir, precision, layers, cfa, tile, window, halo)

writes into out_dir:

    graph.txt      what to allocate and what to dispatch, in order
    NN_name.wgsl   one shader per dispatch
    weights.bin    every weight the shaders read, in their storage type

Kernels (all bind 0..3 read-only, 4 the output; an unused binding gets the
weights buffer; 5 is the window's valid size, see VALID):

    PREP_PACK   mosaic + sigma -> pixel_unshuffle(mosaic, tile) | avg_pool(sigma, tile)
    PREP_SPLIT  the same where a cell's sites change colour (pack_layout in
                common.py): sites split by colour, cell-kind planes, pooled sigma
    CONV_1X1    1x1 conv: 64 px x 32 output channels per workgroup
    CONV_TILED  k x k conv: 16x16 px x 8 output channels per workgroup, the
                input tile and weights staged in workgroup memory
    CONV        plain k x k conv, for layers with fewer than 4 output channels
    HEAD        the full-res head fused: cat[estimate, scatter(m), masks, sigma]
                -> 3x3 (one bank per CFA phase, or one for all repeated per
                phase, folded to 5 input channels)
                -> LeakyReLU -> 1x1, added to the estimate (a residual head) or the
                output itself (DemosaicNet's)

Every conv pads with zeros, as the PyTorch model does (a pixel never sees the
padding from further than the reach, which the halo covers). A conv in a residual
trunk can also apply the LeakyReLU to its input and add its input to its
output (`residual`).
"""
import math
from pathlib import Path

import numpy as np

SHARED_BYTES = 16384     # WebGPU's default workgroup memory; kernels are sized to fit
WG = 8                   # 8x8 workgroups for the simple kernels
HEAD_NPY = (2, 1)        # head pixels per invocation: the first that tiles the window's phase rows
# CONV_TILED: output rows per invocation, and output channels per workgroup for
# layers at least TILED_WIDE channels wide / for narrower ones. Best of 2-3 x 16-32
# on the 5060 Ti (24 MP, ms, old 4 x 8 -> this): X-Trans f16 458 -> 307, f32 520
# -> 365; Bayer 358 -> 262, 505 -> 343 (3 x 32 there: 311); lite and tiny 16-27%
# faster in f16, 7-16% in f32. Output is bitwise the same whatever the shape.
TILED_ROWS = 3
TILED_OCB = (32, 16)
TILED_WIDE = 64


def header(precision):
    if precision == 'f16':
        return 'enable f16;\nalias T = f16;\nalias A = f16;\n'
    return 'alias T = f32;\nalias A = f32;\n'


def subst(src, **kv):
    # longest keys first, so $CINP is replaced before $CIN can match its prefix
    for k in sorted(kv, key=len, reverse=True):
        src = src.replace('$' + k, str(kv[k]).lower() if isinstance(kv[k], bool) else str(kv[k]))
    assert '$' not in src, src
    return src


def ceil(a, b):
    return -(-a // b)


def pow2_at_least(n):
    return 1 << (n - 1).bit_length()


# --------------------------------------------------------------------- kernels

PREP_PACK = r"""
@group(0) @binding(0) var<storage, read> mos: array<T>;
@group(0) @binding(2) var<storage, read> sig: array<T>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const H: u32 = $Hu;    // packed grid
const W: u32 = $Wu;
const TL: u32 = $TLu;  // tile
$VALID
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x;
    let y = g.y;
    if (x >= min(W, valid_w()) || y >= min(H, valid_h())) { return; }
    let plane = H * W;
    let px = y * W + x;
    var s = A(0);
    for (var i = 0u; i < TL; i++) {
        for (var j = 0u; j < TL; j++) {
            let src = (TL * y + i) * TL * W + TL * x + j;
            outp[(i * TL + j) * plane + px] = mos[src];
            s += A(sig[src]);
        }
    }
    outp[TL * TL * plane + px] = T(s / A(TL * TL));
}
"""

# pack_layout's packing (common.py): per cell, each plane is one site's sample,
# kept only in the cell phases q its mask has (it is that colour there); type
# planes are 1 in the phases of their kind of cell. Windows start on whole CFA
# periods, so q from the window's cell coordinates is the image's. $PLANES
# holds one line per plane, from split_code.
PREP_SPLIT = r"""
@group(0) @binding(0) var<storage, read> mos: array<T>;
@group(0) @binding(2) var<storage, read> sig: array<T>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const H: u32 = $Hu;    // packed grid
const W: u32 = $Wu;
const TL: u32 = $TLu;  // tile
const R: u32 = $Ru;    // cells per CFA period, per side
const NP: u32 = $NPu;  // planes before sigma
$VALID
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x;
    let y = g.y;
    if (x >= min(W, valid_w()) || y >= min(H, valid_h())) { return; }
    let plane = H * W;
    let px = y * W + x;
    let src = TL * y * TL * W + TL * x;      // the cell's first site
    let q = (y % R) * R + x % R;             // its phase
$PLANES
    var s = A(0);
    for (var i = 0u; i < TL; i++) {
        for (var j = 0u; j < TL; j++) {
            s += A(sig[src + i * TL * W + j]);
        }
    }
    outp[NP * plane + px] = T(s / A(TL * TL));
}
"""


def split_code(planes, types, tile):
    """PREP_SPLIT's $PLANES, NP: one line per plane of pack_layout."""
    lines = []
    for k, (site, colour, mask) in enumerate(planes):
        v = f'mos[src + {site // tile}u * TL * W + {site % tile}u]'
        if colour is not None:
            v = f'select(T(0), {v}, (({mask}u >> q) & 1u) != 0u)'
        lines.append(f'    outp[{k}u * plane + px] = {v};')
    for k, mask in enumerate(types, len(planes)):
        lines.append(f'    outp[{k}u * plane + px] = select(T(0), T(1), (({mask}u >> q) & 1u) != 0u);')
    return dict(PLANES='\n'.join(lines), NP=len(planes) + len(types))


# Shared by every kernel. Only the first `valid` rows and columns of a window
# hold image data (an edge window's core is shorter); the core needs nothing
# beyond them, as it is `halo` px inside, so work wholly outside is skipped.
# S: window px per px of the kernel's grid.
VALID = r"""
struct Valid { h: u32, w: u32 }
@group(0) @binding(5) var<uniform> valid: Valid;
const S: u32 = $Su;
fn valid_h() -> u32 { return (valid.h + S - 1u) / S; }
fn valid_w() -> u32 { return (valid.w + S - 1u) / S; }
"""

# Shared by the conv kernels: where output channel `oc` of pixel (y, x) goes,
# optionally through pixel_shuffle(., SF).
OUT_INDEX = r"""
fn out_index(oc: u32, y: u32, x: u32) -> u32 {
    if (OUT_SHUF) {
        let s = oc % (SF * SF);
        return ((oc / (SF * SF)) * SF * u32(H) + SF * y + s / SF) * SF * u32(W) + SF * x + s % SF;
    }
    return (oc * u32(H) + y) * u32(W) + x;
}
"""

CONV = r"""
@group(0) @binding(0) var<storage, read> inp: array<T>;
@group(0) @binding(1) var<storage, read> wts: array<T>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const CIN: u32 = $CINu;
const COUT: u32 = $COUTu;
const K: i32 = $K;
const R: i32 = $K / 2;
const H: i32 = $H;
const W: i32 = $W;
const OB: u32 = $OBu;       // output channels per invocation
const WOFF: u32 = $WOFFu;
const BOFF: u32 = $BOFFu;
const OUT_SHUF: bool = $OUT_SHUF;
const SF: u32 = $SFu;
const ACT: bool = $ACT;
$OUT_INDEX$VALID
fn load(c: u32, y0: i32, x0: i32) -> A {
    if (y0 < 0 || y0 >= H || x0 < 0 || x0 >= W) { return A(0); }
    return A(inp[(c * u32(H) + u32(y0)) * u32(W) + u32(x0)]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = i32(g.x);
    let y = i32(g.y);
    if (x >= W || y >= H || g.x >= valid_w() || g.y >= valid_h()) { return; }
    let o0 = g.z * OB;
    let kk = u32(K * K);
    let wb = WOFF + o0 * CIN * kk;
    var acc: array<A, $OB>;
    for (var o = 0u; o < OB; o++) { acc[o] = A(wts[BOFF + o0 + o]); }
    for (var c = 0u; c < CIN; c++) {
        for (var dy = 0; dy < K; dy++) {
            for (var dx = 0; dx < K; dx++) {
                let v = load(c, y + dy - R, x + dx - R);
                let wi = wb + c * kk + u32(dy * K + dx);
                for (var o = 0u; o < OB; o++) {
                    acc[o] += v * A(wts[wi + o * CIN * kk]);
                }
            }
        }
    }
    for (var o = 0u; o < OB; o++) {
        var r = acc[o];
        if (ACT) { r = max(r, r * A(0.1)); }
        outp[out_index(o0 + o, u32(y), u32(x))] = T(r);
    }
}
"""

# Each of the 64 invocations accumulates 4 pixels (one column, rows ly,
# ly+4, ..) x 8 channels in eight vec4 registers, so each staged weight is used
# 4 times and each staged input 8 times. Weights come packed by pack_tiled.
CONV_TILED = r"""
@group(0) @binding(0) var<storage, read> inp: array<T>;
@group(0) @binding(3) var<storage, read> pw: array<vec4<T>>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const CIN: u32 = $CINu;
const CINP: u32 = $CINPu;     // CIN rounded up to CI (packed weights are zero there)
const COUT: u32 = $COUTu;
const NOB: u32 = $NOBu;       // output channels / OCB, rounded up
const K: u32 = $Ku;
const KK: u32 = $Ku * $Ku;
const R: i32 = $K / 2;
const H: i32 = $H;
const W: i32 = $W;
const PWOFF: u32 = $PWOFFu;   // in vec4s
const OUT_SHUF: bool = $OUT_SHUF;
const SF: u32 = $SFu;
const ACT: bool = $ACT;

const TX: u32 = 16u;
const PY: u32 = $PYu;             // output rows per invocation, 4 apart
const TY: u32 = 4u * PY;
const OCB: u32 = $OCBu;           // output channels per workgroup, all in each invocation
const NV: u32 = OCB / 4u;
const CI: u32 = $CIu;             // input channels staged per chunk
const SW: u32 = TX + K - 1u;
const SH: u32 = TY + K - 1u;
const SP: u32 = SH * SW;          // one staged input plane
const NWC: u32 = CI * KK * NV;    // staged weight vec4s per chunk

var<workgroup> sx: array<T, $NX>;
var<workgroup> sw: array<vec4<T>, $NW>;
$OUT_INDEX$VALID
fn load(c: u32, y0: i32, x0: i32) -> T {
    if (c >= CIN || y0 < 0 || y0 >= H || x0 < 0 || x0 >= W) { return T(0); }
    let y = y0;
    let x = x0;
    return inp[(c * u32(H) + u32(y)) * u32(W) + u32(x)];
}

fn store(oc0: u32, y: i32, x: i32, a: vec4<A>) {
    for (var o = 0u; o < 4u; o++) {
        let oc = oc0 + o;
        if (oc >= COUT) { break; }
        var r = a[o];
        if (ACT) { r = max(r, r * A(0.1)); }
        outp[out_index(oc, u32(y), u32(x))] = T(r);
    }
}

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    if (wg.x * TX >= valid_w() || wg.y * TY >= valid_h()) { return; }
    let ox0 = i32(wg.x * TX);
    let oy0 = i32(wg.y * TY);
    let ob = wg.z;
    let x = ox0 + i32(l.x);
    let y0 = oy0 + i32(l.y);

$INIT

    for (var c0 = 0u; c0 < CINP; c0 += CI) {
        for (var i = li; i < SP; i += 64u) {
            let gy = oy0 + i32(i / SW) - R;
            let gx = ox0 + i32(i % SW) - R;
            for (var ci = 0u; ci < CI; ci++) {
                sx[ci * SP + i] = load(c0 + ci, gy, gx);
            }
        }
        for (var i = li; i < NWC; i += 64u) {
            sw[i] = pw[PWOFF + (ob * CINP + c0) * KK * NV + i];
        }
        workgroupBarrier();
        for (var ci = 0u; ci < CI; ci++) {
            for (var ky = 0u; ky < K; ky++) {
                for (var kx = 0u; kx < K; kx++) {
                    let s = (ci * SH + l.y + ky) * SW + l.x + kx;
                    let wt = (ci * KK + ky * K + kx) * NV;
$TAP
                }
            }
        }
        workgroupBarrier();
    }

    if (x >= W) { return; }
    var accs = array<vec4<A>, $NACC>($ACCS);
    for (var p = 0u; p < PY; p++) {
        let y = y0 + i32(4u * p);
        if (y >= H) { continue; }
        for (var h = 0u; h < NV; h++) {
            let oc0 = ob * OCB + 4u * h;
            let bo = PWOFF + NOB * CINP * KK * NV + oc0 / 4u;
            store(oc0, y, x, accs[NV * p + h] + vec4<A>(pw[bo]));
        }
    }
}
"""

# A workgroup covers 64 pixels x 32 output channels, so the input is re-read
# COUT/32 times; each invocation accumulates 4 pixels (16 apart, so loads
# coalesce) x 8 channels. Pixels are addressed linearly; the dispatch is 2-D
# only to stay under 65535 workgroups per dimension. Weights packed by pack_1x1.
CONV_1X1 = r"""
@group(0) @binding(0) var<storage, read> inp: array<T>;
@group(0) @binding(3) var<storage, read> pw: array<vec4<T>>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;

const CIN: u32 = $CINu;
const CINP: u32 = $CINPu;
const COUT: u32 = $COUTu;
const NOG: u32 = $NOGu;       // output-channel groups of OCW
const H: u32 = $Hu;
const W: u32 = $Wu;
const NGX: u32 = $NGXu;       // workgroups along x of the 2-D dispatch
const PWOFF: u32 = $PWOFFu;   // in vec4s
const OUT_SHUF: bool = $OUT_SHUF;
const SF: u32 = $SFu;
const ACT: bool = $ACT;

const NP: u32 = 64u;
const OCW: u32 = 32u;
const CI: u32 = $CIu;

var<workgroup> sx: array<T, $NX>;           // [ci][pixel]
var<workgroup> sw: array<vec4<T>, $NW>;     // [ci][8 vec4 = 32 channels]
$OUT_INDEX$VALID
fn load(c: u32, p: u32) -> T {
    if (c >= CIN || p >= H * W) { return T(0); }
    return inp[c * H * W + p];
}

fn store(oc0: u32, p: u32, a: vec4<A>) {
    if (p >= H * W) { return; }
    for (var o = 0u; o < 4u; o++) {
        let oc = oc0 + o;
        if (oc >= COUT) { break; }
        var r = a[o];
        if (ACT) { r = max(r, r * A(0.1)); }
        outp[out_index(oc, p / W, p % W)] = T(r);
    }
}

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let p0 = (wg.y * NGX + wg.x) * NP;
    // the NP pixels run along a row, possibly into the next: skipped if they
    // start below the valid rows, or right of the valid columns and end there
    // or on a row below them
    let py = p0 / W;
    let px0 = p0 % W;
    if (py >= valid_h() || (px0 >= valid_w() && (px0 + NP <= W || py + 1u >= valid_h()))) {
        return;
    }
    let og = wg.z;
    var a00 = vec4<A>(0); var a01 = vec4<A>(0);
    var a10 = vec4<A>(0); var a11 = vec4<A>(0);
    var a20 = vec4<A>(0); var a21 = vec4<A>(0);
    var a30 = vec4<A>(0); var a31 = vec4<A>(0);
    let wo = l.y * 2u;           // this invocation's 8 channels = vec4s wo, wo+1
    for (var c0 = 0u; c0 < CINP; c0 += CI) {
        for (var i = li; i < CI * NP; i += 64u) {
            sx[i] = load(c0 + i / NP, p0 + i % NP);
        }
        for (var i = li; i < CI * 8u; i += 64u) {
            sw[i] = pw[PWOFF + (og * CINP + c0) * 8u + i];
        }
        workgroupBarrier();
        for (var ci = 0u; ci < CI; ci++) {
            let s = ci * NP + l.x;
            let v0 = A(sx[s]);
            let v1 = A(sx[s + 16u]);
            let v2 = A(sx[s + 32u]);
            let v3 = A(sx[s + 48u]);
            let w0 = vec4<A>(sw[ci * 8u + wo]);
            let w1 = vec4<A>(sw[ci * 8u + wo + 1u]);
            a00 += v0 * w0; a01 += v0 * w1;
            a10 += v1 * w0; a11 += v1 * w1;
            a20 += v2 * w0; a21 += v2 * w1;
            a30 += v3 * w0; a31 += v3 * w1;
        }
        workgroupBarrier();
    }
    let oc0 = og * OCW + l.y * 8u;
    let bo = PWOFF + NOG * CINP * 8u + oc0 / 4u;
    let b0 = vec4<A>(pw[bo]);
    let b1 = vec4<A>(pw[bo + 1u]);
    let px = p0 + l.x;
    store(oc0, px, a00 + b0);       store(oc0 + 4u, px, a01 + b1);
    store(oc0, px + 16u, a10 + b0); store(oc0 + 4u, px + 16u, a11 + b1);
    store(oc0, px + 32u, a20 + b0); store(oc0 + 4u, px + 32u, a21 + b1);
    store(oc0, px + 48u, a30 + b0); store(oc0 + 4u, px + 48u, a31 + b1);
}
"""

def tiled_code(py, nv):
    """The unrolled parts of CONV_TILED for py rows x nv vec4s of output
    channels per invocation: accumulators a<p>_<j>, the loads and FMAs of one
    tap, and the accumulators in store order."""
    acc = [f'a{p}_{j}' for p in range(py) for j in range(nv)]
    init = '\n'.join(f'    var {a} = vec4<A>(0);' for a in acc)
    tap = ([f'                    let v{p} = A(sx[s + {4 * p}u * SW]);' for p in range(py)] +
           [f'                    let w{j} = vec4<A>(sw[wt + {j}u]);' for j in range(nv)] +
           [f'                    a{p}_{j} += v{p} * w{j};' for p in range(py) for j in range(nv)])
    return dict(INIT=init, TAP='\n'.join(tap), NACC=len(acc), ACCS=', '.join(acc))


# The head's 3x3 sees [estimate, scatter(m), masks, sigma] (10 channels), but
# within one CFA phase the masks are constants and each tap's scattered mosaic
# is nonzero in one known plane, so `fold_head` turns every phase bank into a
# 3x3 on [estimate, m, sigma] (5 channels) whose bias holds the masks' part.
# (Taps beyond the window's edge read zeros, but their masks stay in the bias,
# where PyTorch's zero padding has none: the window's outer ring, which the
# halo keeps out of every core.)
# Each workgroup covers the pixels of one phase (8 x 8*NPY of them, P px apart)
# and stages that bank's weights once; each invocation keeps NPY pixels' hidden
# channels in vec4 registers, reusing every weight vec4 it loads NPY times.
# A residual head's output is added to the estimate (RESID); otherwise it is the image.
HEAD = r"""
@group(0) @binding(0) var<storage, read> up: array<T>;     // 3 x H x W estimate
@group(0) @binding(1) var<storage, read> pw: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> mos: array<T>;
@group(0) @binding(3) var<storage, read> sig: array<T>;
@group(0) @binding(4) var<storage, read_write> outp: array<T>;  // 3 x H x W: estimate + head

const H: u32 = $Hu;
const W: u32 = $Wu;
const P: u32 = $Pu;
const NV: u32 = $NVu;       // head width / 4
const NPY: u32 = $NPYu;      // pixels per invocation, 8 phase rows apart
const PHASE_OUT: bool = $PHASE_OUT;  // final 1x1 has one bank per CFA phase
const W0: u32 = $W0u;      // per bank: [tap][estimate, m, sigma][NV vec4], then NV vec4 bias
const NW0: u32 = $NW0u;
const W1: u32 = $W1u;      // per bank: [rgb][NV vec4], then one vec4 bias
const NW1: u32 = $NW1u;
var<workgroup> sw: array<vec4<T>, NW0>;
var<workgroup> so: array<vec4<T>, NW1>;
$VALID
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let bank = wg.z;
    if (wg.x * 8u * P + bank % P >= valid_w() || wg.y * NPY * 8u * P + bank / P >= valid_h()) {
        return;
    }
    let bo = select(0u, bank, PHASE_OUT);
    for (var i = li; i < NW0; i += 64u) { sw[i] = pw[W0 + bank * NW0 + i]; }
    for (var i = li; i < NW1; i += 64u) { so[i] = pw[W1 + bo * NW1 + i]; }
    workgroupBarrier();
    let x = (wg.x * 8u + l.x) * P + bank % P;
    if (x >= W) { return; }
    var y: array<u32, NPY>;
    for (var p = 0u; p < NPY; p++) {
        // past the bottom: computed on a wrapped row, not stored
        y[p] = (((wg.y * NPY + p) * 8u + l.y) * P + bank / P) % H;
    }
$INIT
    for (var dy = 0u; dy < 3u; dy++) {
        for (var dx = 0u; dx < 3u; dx++) {
            let xx = x + dx - 1u;     // past the left edge it wraps to >= W: outside, like the right
            let wi = (dy * 3u + dx) * 5u * NV;
$TAP
        }
    }
    let plane = H * W;
    let rb = vec4<A>(so[3u * NV]);
    for (var p = 0u; p < NPY; p++) {
        if (((wg.y * NPY + p) * 8u + l.y) * P + bank / P >= H) { break; }
        let px = y[p] * W + x;
        for (var k = 0u; k < 3u; k++) {
            var r = rb[k]$RESID;
$OUT
            outp[k * plane + px] = T(r);
        }
    }
}
"""


def head_code(nv, npy):
    """The unrolled parts of HEAD: accumulators h<p>_<j> (pixel p, hidden
    channels 4j..4j+3) start at the bias, take one 3x3 tap, and are projected
    to RGB."""
    init = [f'    var h{p}_{j} = vec4<A>(sw[{9 * 5 * nv + j}u]);'
            for p in range(npy) for j in range(nv)]
    tap = []
    for p in range(npy):
        tap.append(f'            let yy{p} = y[{p}] + dy - 1u;')
        tap.append(f'            let ok{p} = xx < W && yy{p} < H;')
        tap.append(f'            let i{p} = select(0u, yy{p} * W + xx, ok{p});')
        tap.append(f'            let g{p} = select(A(0), A(1), ok{p});')
        tap.append(f'            let v{p} = array<A, 5>(g{p} * A(up[i{p}]), g{p} * A(up[H * W + i{p}]), '
                   f'g{p} * A(up[2u * H * W + i{p}]), g{p} * A(mos[i{p}]), g{p} * A(sig[i{p}]));')
    for c in range(5):
        for j in range(nv):
            tap.append(f'            let w{c}_{j} = vec4<A>(sw[wi + {c * nv + j}u]);')
        for p in range(npy):
            tap += [f'            h{p}_{j} += v{p}[{c}] * w{c}_{j};' for j in range(nv)]
    out = []
    for p in range(npy):
        out.append(f'            {"if" if p == 0 else "} else if"} (p == {p}u) {{')
        out += [f'                r += dot(max(h{p}_{j}, h{p}_{j} * A(0.1)), '
                f'vec4<A>(so[k * NV + {j}u]));' for j in range(nv)]
    out.append('            }')
    return dict(INIT='\n'.join(init), TAP='\n'.join(tap), OUT='\n'.join(out))


# -------------------------------------------------------------- weight packing

class Weights:
    """The one weights buffer. `raw` appends an array as-is and returns its
    element offset; `vec4` appends a packed array starting on a vec4 boundary
    and returns its offset in vec4s (what pw[] indexes)."""

    def __init__(self):
        self.parts, self.n = [], 0

    def raw(self, a):
        a = np.asarray(a, np.float32).ravel()
        self.parts.append(a)
        self.n += a.size
        return self.n - a.size

    def vec4(self, a):
        pad = -self.n % 4
        if pad:
            self.raw(np.zeros(pad, np.float32))
        return self.raw(a) // 4


def pack_tiled(w, b, ci, ocb):
    """(cout, cin, k, k) -> [cout/ocb][cin padded to ci][tap][ocb], then the
    biases padded to a multiple of ocb. Padding is zeros."""
    cout, cin, k, _ = w.shape
    cinp, coutp = ceil(cin, ci) * ci, ceil(cout, ocb) * ocb
    p = np.zeros((coutp, cinp, k * k), np.float32)
    p[:cout, :cin] = w.reshape(cout, cin, k * k)
    p = p.reshape(coutp // ocb, ocb, cinp, k * k).transpose(0, 2, 3, 1)
    return np.concatenate([p.ravel(), np.pad(b, (0, coutp - cout))])


def pack_1x1(w, b, ci):
    """(cout, cin, 1, 1) -> [cout/32][cin padded to ci][32], then the biases
    padded to a multiple of 32."""
    cout, cin = w.shape[:2]
    cinp, coutp = ceil(cin, ci) * ci, ceil(cout, 32) * 32
    p = np.zeros((coutp, cinp), np.float32)
    p[:cout, :cin] = w[:, :, 0, 0]
    p = p.reshape(coutp // 32, 32, cinp).transpose(0, 2, 1)
    return np.concatenate([p.ravel(), np.pad(b, (0, coutp - cout))])


def fold_head(w, b, banks, cfa):
    """The head's 3x3, (banks * width, 10, 3, 3) on [estimate, scatter(m),
    masks, sigma] -> per CFA phase q = (y % P) * P + x % P: [tap][estimate, m,
    sigma][width], then the width biases with the masks' constant part added.
    A single bank is repeated for every phase."""
    period = cfa.shape[0]
    w = w.astype(np.float64).reshape(banks, -1, 10, 3, 3)
    b = b.astype(np.float64).reshape(banks, -1)
    if banks == 1:
        w, b = np.repeat(w, period ** 2, 0), np.repeat(b, period ** 2, 0)
    out = []
    for q in range(period ** 2):
        py, px = divmod(q, period)
        f = np.zeros((9, 5, w.shape[1]))
        bias = b[q].copy()
        for dy in range(3):
            for dx in range(3):
                col = int(cfa[(py + dy - 1) % period, (px + dx - 1) % period])
                t = w[q, :, :, dy, dx]                       # width, 10
                f[dy * 3 + dx] = np.stack([t[:, 0], t[:, 1], t[:, 2], t[:, 3 + col], t[:, 9]])
                bias += t[:, 6 + col]
        out += [f.ravel(), bias]
    return np.concatenate(out)


def pack_head_out(w, b):
    """The head's 1x1, (banks * 3, width, 1, 1) -> per bank: [rgb][width], then
    the 3 biases padded to a vec4."""
    w, b = w.reshape(-1, 3, w.shape[1]), b.reshape(-1, 3)
    return np.concatenate([np.concatenate([wb.ravel(), bb, [0]]) for wb, bb in zip(w, b)])


# --------------------------------------------------------- residual layers

# Spliced into CONV_TILED / CONV_1X1 (each anchor once) for a residual trunk's
# layers, so kernels without them stay as they were: the LeakyReLU on the
# input as it is loaded, and the input's own channel at the output pixel added
# before the store (the conv maps C channels to C).
RESIDUAL = {
    'tiled': ('    return inp[(c * u32(H) + u32(y)) * u32(W) + u32(x)];\n',
              '        r += A(inp[(oc * u32(H) + u32(y)) * u32(W) + u32(x)]);\n'),
    '1x1': ('    return inp[c * H * W + p];\n',
            '        r += A(inp[oc * H * W + p]);\n'),
}
STORE = '        var r = a[o];\n'


def residual(src, kind, act_in, skip):
    load, add = RESIDUAL[kind]
    assert src.count(load) == 1 and src.count(STORE) == 1
    if act_in:
        src = src.replace(load, load.replace('return ', 'let v = ') +
                          '    return max(v, v * T(0.1));\n')
    if skip:
        src = src.replace(STORE, STORE + add)
    return src


# ---------------------------------------------------------------------- graph

class Graph:
    """Ops on logical tensors; `plan` then packs the tensors into as few
    buffers as their lifetimes allow (no op ever reads and writes one buffer)."""

    def __init__(self, precision, window):
        self.precision, self.window = precision, window
        self.elem = 2 if precision == 'f16' else 4
        self.weights = Weights()
        self.tensors = []        # [elems, defining op (-1 = input), last reading op]
        self.ops = []            # (name, wgsl, [binding 0..4 tensor or None], groups)

    def tensor(self, elems):
        self.tensors.append([elems, -1, -1])
        return len(self.tensors) - 1

    def op(self, name, wgsl, binds, groups):
        i = len(self.ops)
        for t in binds[:4]:
            if t is not None:
                self.tensors[t][2] = i
        self.tensors[binds[4]][1] = self.tensors[binds[4]][2] = i
        self.ops.append((name, header(self.precision) + wgsl, binds, groups))

    def conv(self, name, x, w, b, h, wd, act, shuffle=1, act_in=False, skip=False):
        """k x k conv (zero padded) on the h x wd grid; shuffle > 1 writes
        pixel_shuffle(., shuffle) of the result. act_in: LeakyReLU on the
        input; skip: the input added to the output (see residual)."""
        cout, cin, k, _ = w.shape
        assert not skip or (cin == cout and shuffle == 1)
        out = self.tensor(cout * h * wd)
        flags = dict(CIN=cin, COUT=cout, H=h, W=wd, ACT=act, OUT_SHUF=shuffle > 1,
                     SF=shuffle, OUT_INDEX=OUT_INDEX, VALID=VALID, S=self.window // h)
        if k == 1:
            ci = 32
            while ci > 1 and self.elem * ci * (64 + 32) > SHARED_BYTES:
                ci //= 2
            ci = min(ci, pow2_at_least(cin))
            off = self.weights.vec4(pack_1x1(w, b, ci))
            nwg = ceil(h * wd, 64)
            ngx = min(nwg, 32768)
            src = subst(CONV_1X1, CINP=ceil(cin, ci) * ci, NOG=ceil(cout, 32), NGX=ngx,
                        PWOFF=off, CI=ci, NX=ci * 64, NW=ci * 8, **flags)
            src = residual(src, '1x1', act_in, skip)
            self.op(name, src, [x, None, None, None, out], (ngx, ceil(nwg, ngx), ceil(cout, 32)))
        elif cout >= 4:
            py, ocb = TILED_ROWS, TILED_OCB[0] if cout >= TILED_WIDE else TILED_OCB[1]
            ocb = min(ocb, ceil(cout, 4) * 4)       # no wider than the layer
            sw, sh = 16 + k - 1, 4 * py + k - 1
            ci = 16
            while ci > 1 and self.elem * ci * (sh * sw + ocb * k * k) > SHARED_BYTES:
                ci //= 2
            ci = min(ci, pow2_at_least(cin))
            off = self.weights.vec4(pack_tiled(w, b, ci, ocb))
            src = subst(CONV_TILED, CINP=ceil(cin, ci) * ci, NOB=ceil(cout, ocb), K=k,
                        PWOFF=off, CI=ci, NX=ci * sh * sw, NW=ci * k * k * ocb // 4, PY=py,
                        OCB=ocb, **tiled_code(py, ocb // 4), **flags)
            src = residual(src, 'tiled', act_in, skip)
            self.op(name, src, [x, None, None, None, out],
                    (ceil(wd, 16), ceil(h, 4 * py), ceil(cout, ocb)))
        else:
            assert not act_in and not skip
            ob = next(d for d in (8, 4, 3, 2, 1) if cout % d == 0)
            src = subst(CONV, K=k, OB=ob, WOFF=self.weights.raw(w), BOFF=self.weights.raw(b), **flags)
            self.op(name, src, [x, None, None, None, out], (ceil(wd, WG), ceil(h, WG), cout // ob))
        return out

    def plan(self):
        """Greedy, in definition order: a tensor takes the smallest free buffer
        that fits, else the largest free one (grown), else a new one. A buffer
        is free once the last op reading it has run."""
        slots, where = [], {}                # slots: [bytes, last use]
        for t in sorted(range(len(self.tensors)), key=lambda t: self.tensors[t][1]):
            elems, d, last = self.tensors[t]
            nbytes = ceil(elems * self.elem, 4) * 4
            free = [i for i, s in enumerate(slots) if s[1] < d]
            fit = [i for i in free if slots[i][0] >= nbytes]
            if fit:
                i = min(fit, key=lambda i: slots[i][0])
            elif free:
                i = max(free, key=lambda i: slots[i][0])
            else:
                slots.append([0, 0])
                i = len(slots) - 1
            slots[i] = [max(slots[i][0], nbytes), last]
            where[t] = i + 1                 # buffer 0 is the weights
        return [s[0] for s in slots], where


def write(out_dir, precision, layers, cfa, tile, window, halo):
    """layers: the Demosaic as plain arrays:
        trunk      [(name, weight (cout, cin, k, k), bias, leaky relu after?,
                     leaky relu on the input?, input added to the output?)]
        pack       common.pack_layout(cfa, tile)
        head_in    (weight, bias, banks): the head's 3x3, banks = 1 or period^2,
                   weight (banks * width, 10, 3, 3); None without a head
        head_out   (weight, bias, banks): its 1x1 to RGB, (banks * 3, width, 1, 1)
        head_residual  whether the head is added to the estimate (default) or is the output
    The last trunk conv's output, pixel-shuffled to full resolution, is the
    estimate (3 x h x w), and the output with the head added if there is one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob('*.wgsl'):            # a previous export's ops, which may differ
        old.unlink()
    period = cfa.shape[0]
    h = w = window
    hp, wp = h // tile, w // tile
    g = Graph(precision, h)
    mos, sig = g.tensor(h * w), g.tensor(h * w)

    planes, types, r = layers['pack']
    if r == 1:
        x = g.tensor((tile * tile + 1) * hp * wp)
        src = subst(PREP_PACK, H=hp, W=wp, TL=tile, VALID=VALID, S=tile)
    else:
        n = len(planes) + len(types)
        x = g.tensor((n + 1) * hp * wp)
        src = subst(PREP_SPLIT, H=hp, W=wp, TL=tile, R=r, VALID=VALID, S=tile,
                    **split_code(planes, types, tile))
    g.op('prep', src, [mos, None, sig, None, x], (ceil(wp, WG), ceil(hp, WG), 1))
    for i, (name, wt, b, act, act_in, skip) in enumerate(layers['trunk']):
        last = i == len(layers['trunk']) - 1
        x = g.conv(name, x, wt, b, hp, wp, act, shuffle=tile if last else 1,
                   act_in=act_in, skip=skip)

    if layers.get('head_in') is None:
        out = x                               # the last conv wrote 3 x h x w
    else:
        (w0, b0, n_in), (w1, b1, n_out) = layers['head_in'], layers['head_out']
        out = g.tensor(3 * h * w)
        width = w0.shape[0] // n_in
        assert width % 4 == 0 and h % period == 0 and w % period == 0
        nv, rows = width // 4, ceil(h, period)
        npy = next(n for n in HEAD_NPY if rows % (8 * n) == 0 or n == 1)
        src = subst(HEAD, H=h, W=w, P=period, NV=nv, NPY=npy, PHASE_OUT=n_out > 1,
                    W0=g.weights.vec4(fold_head(w0, b0, n_in, cfa)), NW0=(9 * 5 + 1) * nv,
                    W1=g.weights.vec4(pack_head_out(w1, b1)), NW1=3 * nv + 1,
                    VALID=VALID, S=1,
                    RESID=' + A(up[k * plane + px])' if layers.get('head_residual', True) else '',
                    **head_code(nv, npy))
        groups = (ceil(ceil(w, period), 8), ceil(rows, 8 * npy), period * period)
        g.op('head', src, [x, None, mos, sig, out], groups)
    # the output outlives every op, and so do the inputs: wgpu-run's place
    # shader reads the mosaic and sigma after the graph, for the grain (without
    # a head nothing but the first op reads them, and the output would take the
    # mosaic's buffer)
    for t in (out, mos, sig):
        g.tensors[t][2] = len(g.ops)

    sizes, where = g.plan()
    weights = np.concatenate(g.weights.parts)
    weights = weights.astype('<f2' if precision == 'f16' else '<f4')
    weights.tofile(out_dir / 'weights.bin')
    lines = ['# generated by training/common.py export(); executed by wgpu-run',
             f'storage {precision}',
             f'window {h} {w}',
             f'halo {halo}',
             f'multiple {math.lcm(tile, period)}',
             f'cfa {period} ' + ' '.join(str(int(c)) for c in cfa.ravel()),
             f'buffer 0 {ceil(weights.nbytes, 4) * 4} weights.bin']
    lines += [f'buffer {i + 1} {s}' for i, s in enumerate(sizes)]
    lines += [f'mosaic {where[mos]}', f'sigma {where[sig]}', f'output {where[out]}']
    for i, (name, src, binds, groups) in enumerate(g.ops):
        fname = f'{i:02d}_{name}.wgsl'
        (out_dir / fname).write_text(src)
        ids = [0 if t is None else where[t] for t in binds]
        lines.append(f'op {fname} ' + ' '.join(map(str, ids + list(groups))))
    (out_dir / 'graph.txt').write_text('\n'.join(lines) + '\n')
    return len(g.ops), sum(sizes)
