# AGENTS.md

Starfish neural demosaic: PyTorch joint demosaic + denoise networks for X-Trans and
Bayer, compiled to WGSL shaders that the Starfish app (a separate repo,
`../starfish`, with `wgpu-run/`, the model-agnostic executor) loads. This repo
was split from the app repo in 2026-11 (v0.2.0); history before then is the
app repo's `training/` folder, kept here. Notes below were written when the two
were one: a path written `training/x` is `x` here; "the app", `src/`,
`wgpu-run/`, `web/` and `build.rs` are in the app repo (see its AGENTS.md,
"How the app uses the models"). `demosaic-run` is built there:
`cargo build --release -p demosaic-run` and run it on an export here, e.g.
`../starfish/target/release/demosaic-run xtrans_wgpu`.

**Releasing a model to the app:** export here (`--export`), check it against
test.bin with demosaic-run, then copy the `*_wgpu/{f16,f32}/` folders (not
test.bin) into the app repo's `models/`, which is what the app embeds and
debug builds read (`../starfish/scripts/sync-models.sh` does it).

**History:** this repo starts at v0.2.0. Where these notes say something is
"in git history" (removed code, older models, earlier scripts), they mean the
development history before v0.2.0, which was not published. Their descriptions
here are what is left of it.

## Layout

```
(repo root)    PyTorch joint demosaic + denoise for X-Trans and Bayer
  common.py        everything shared: data constants, CFA tables, network parts,
                   the synthetic sensor, loss, training loop, export, inference
  train_xtrans.py  X-Trans model: training constants + build() with every layer written out
  train_bayer.py   the same for Bayer (see "The model")
  train_*_res.py   the residual networks with DemosaicNet's head (xtrans_res.pt, bayer_res.pt; see "The model")
  train_*_lite.py  the same at half the width, for weaker GPUs
  train_*_tiny.py  lite at half the depth, for the weakest
  infer_*.py       mosaic + noise a clean photo, demosaic it, print PSNR (*_lite, *_tiny: those models)
  finetune.py      fine-tunes a trained model with today's training into <name>_ft.pt
  modal_train.py   runs a train_*.py (or "finetune.py train_x") on a Modal GPU, data and checkpoints on the `starfish` volume
  suite.py         scores trained models per artifact category -> training/suite/
  wgsl.py          compiles a trained model into WGSL shaders + graph.txt
  prep.py          data/DIV2K_train_HR/*.png -> data/div2k.npy + data/div2k_index.npy
  requirements.txt the Python packages (torch cu128, numpy, pillow), for training/.venv
  data/            training data, added by hand (README.md says what; the rest is gitignored)
  xtrans.pt, bayer.pt           the trained models
  xtrans_wgpu/, bayer_wgpu/     their exports, which the app loads
```

Design rules for training/: no abstraction beyond plain PyTorch; every model
and training choice is a constant at the top of a file, never a command-line
flag; checkpoints are plain `torch.save` dicts; no checkpoint validation
beyond what `load_state_dict` does. Keep it that way.

## Commands

The DIV2K training images go in `training/data/DIV2K_train_HR/`, added by
hand (`training/data/README.md`); `prep.py` turns them into
`training/data/div2k.npy` + `div2k_index.npy`, the default `DATA` of training
and export (`--data` still overrides it). That folder is gitignored except its
README. The Python venv is `training/.venv` (gitignored; Python 3.14, the
packages in `training/requirements.txt`: torch 2.11 cu128, as the GPU is
Blackwell sm_120 and needs a CUDA 12.8+ build; everything assumes `cuda`).
Its install command is at the top of `requirements.txt`. Test raws and
`images/` live outside the repo, in `../demosaic/`, which also holds a
prepared `div2k.npy` + `div2k_index.npy` (`--data ../../demosaic/div2k.npy`
instead of running `prep.py`). From `training/`:

```
.venv/bin/python prep.py   # once, after adding data/DIV2K_train_HR/
.venv/bin/python train_xtrans.py   # resumes xtrans.pt
.venv/bin/python train_xtrans.py --export   # -> xtrans_wgpu/
.venv/bin/python infer_xtrans.py ../../demosaic/images/kodim21.png --a 1e-3
```

Demosaic-run (from the app repo) checks an export: see the app's AGENTS.md "Commands".
`test.bin` is gitignored; it is rewritten by every export.

## The model

**`train_xtrans_res.py` / `train_bayer_res.py` (2026-10-25, not yet trained)**
are a5cd8e3's residual network with a wider second half and DemosaicNet's
head, beside the plain models below (their checkpoints `xtrans.pt` /
`bayer.pt` and exports are untouched; `xtrans_res.pt` -> `xtrans_res_wgpu/`,
and `suite.py train_xtrans train_xtrans_res` scores both in one run).
- Trunk: pre-activation `Residual` blocks. First half at `WIDTH_1`
  (stem + blocks, `LAYERS_1` 3x3s), then one 3x3 that widens the stream (the
  LeakyReLU before it is the last block's), blocks at `WIDTH_2`, and the
  output 3x3 (`LAYERS_2` 3x3s). X-Trans tile 3: 64 wide x 7, 128 wide x 4
  (11 3x3s). Bayer tile 2: 32 wide x 9, 64 wide x 6 (15 3x3s).
- Head: `common.head(.., residual=False)`, `HEAD_RESIDUAL = False`: the
  shared 3x3 at 64 on [estimate, scattered mosaic, masks, sigma] and a 1x1,
  whose output is the image (DemosaicNet; zero-start and added to the
  estimate only with `residual`). DemosaicNet's own head sees the mosaic and
  the upsampled trunk only (checked against mgharbi/demosaicnet's
  `modules.py`: 15 plain 3x3s at 64, a half-resolution split-and-multiply,
  12 channels, a grouped 2x2 transposed conv, a 3x3 to 64 and a 1x1);
  masks are kept because the weights are shared across 36 phases, sigma
  because the head can then follow the noise. `wgsl.py` takes it as
  `layers['head_residual']` (`Demosaic.head_residual`).
- Reach 36 px (X-Trans, `HALO` 36) / 32 (Bayer, `HALO` 32), each trunk 3x3
  one cell and the head's one px, measured by gradients; parameters 636k /
  255k. `NOISE = None`: common's full range (a 1e-5..1e-2, b 1e-8..1e-4,
  the app's clamps), `STEPS` 1M, patch 192 (the loss
  drops 36 / 32 px per side: 39% / 44% of the pixels count).
- Exports against PyTorch (random weights): f32 90.6 / 91.0 dB, f16 74.9 /
  73.7. 24 MP `--gpu --bench` (these are the 42 / 34 px halo versions, one
  more block each): X-Trans f16 580-1030 ms, Bayer 511-520, against the
  plain models' 264 / 278. The X-Trans runs varied by 2x while the plain
  export stayed at 264 ms, with the same clocks (2730 MHz) and lower power
  when slow; not explained (VS Code's and darktable's GPU use was listed),
  not tried again.

**Since 2026-10-23 `train_xtrans.py` and `train_bayer.py` are a plain trunk
and a shared full-resolution head, for low ISO; the rest of this section
(residual blocks, the per-phase head, circular padding) describes the
networks before, which `train_*_lite.py` / `_tiny.py` still are.** A
separate, larger model for high ISO is planned. The new ones:

- Trunk: no residual blocks, `LAYERS` 3x3s at `WIDTH` with LeakyReLU, then
  the output 3x3, whose `pixel_shuffle` is the estimate. X-Trans tile 3,
  width 128 (96 until 2026-10-23: 128 brings it to Bayer's runtime), 4 3x3s; Bayer tile 2, width 64, 6 3x3s. Same packing and CFA
  input planes as before; zero padding (see "The wgpu export"). A 3x3
  costs 9 (W / tile)^2 per px, the same in both (9216); X-Trans has fewer
  because each reaches 3 px. X-Trans `MIX` (off) puts a 1x1 after each
  inner 3x3. 24 MP f16 / f32 with random weights (2026-10-23, same
  session), training fwd + loss + bwd per batch of 32: X-Trans 96 167 /
  201 ms, 13.5 ms; 96 + MIX 184 / 246, 14.9; 128 266 / 308, 15.4; 128 +
  MIX 307 / 392, 17.7; Bayer 281 / 309, 18.2. A 1x1 at 128 takes ~20 /
  ~42 ms against ~106 / ~122 for an inner 3x3: a ninth of the arithmetic at
  ~60% (f16) / ~35% (f32) of the 3x3's efficiency. Dispatching `CONV_1X1`'s
  output-channel groups fastest (so the four that read one pixel block run
  together) changed nothing (bitwise the same output, within 1-2%): it is
  not the input re-reads. Its threads do 4 px x 8 channels, ~5 FMAs per
  workgroup-memory load, as `CONV_TILED` did before its shape was tuned.
- Head (`common.head`, `HEAD_WIDTH` 64): a 3x3 on [estimate, scattered
  mosaic, masks, sigma] shared by every CFA phase (`HEAD_PER_PHASE` False:
  it finds the phase from the masks and samples around the pixel), a
  LeakyReLU and a 1x1 to RGB from zero, added to the estimate. The trunk
  ends in a linear conv shared by the cell's positions; the head is a
  nonlinear network at every pixel. Every 3x3 window of either CFA holds
  every colour. The export folds the one bank per phase like a per-phase
  head, so it costs the same.
- Reach: X-Trans 14 + 1 = 15 px, `HALO` 18 (a fourth trunk 3x3 would fit
  the same halo); Bayer 13 + 1 = 14, `HALO` 14. Checked by gradients.
- No CA input, and no CA in training since 2026-10-25 (see "How the app
  uses the models"). `STEPS` 500k.
- Parameters 350k (X-Trans) / 164k (Bayer). Batch 32 on the 5060 Ti:
  forward + loss + backward 15.4 / 18.2 ms; a real X-Trans step with the
  `CA_FIX` split of then 21.5 ms (~1490 img/s), so ~3.0 h for 500k steps (Bayer, not
  timed, ~3 ms more). 24 MP `--gpu --bench` f16 / f32:
  X-Trans 266 / 308 ms, Bayer 281 / 309 (the head-free plain models
  before: 132 / 268 f16). Export vs PyTorch (2000-step weights, X-Trans
  at 96 wide): f32 90 / 87 dB, f16 60 / 58.
- Not yet trained. The checkpoints `xtrans.pt` / `bayer.pt` in the folder
  are still the head-free plain models of 2026-10-18 (with `CA = True`),
  which do not load into these: move them aside before training.

The plain models of 2026-10-18 (no head, `LAYERS` 5 at width 64 for both,
X-Trans reach 20 px, `CA = True`, 174k / 160k parameters) were trained on `NOISE`, a log-uniform over 1e-6..1e-4
(about ISO 800 on an APS-C sensor, 1e-4, down to 50 dB at mid-grey) and b
over 1e-10..1e-6, plus the 20% with no noise at all: a per-model constant
that `train` / `export` / `eval_set` / `random_batch` take as `noise`
(`finetune.py` reads `NOISE`), the export's test vector at the ranges'
geometric middles. The app's noise fit and sliders can go beyond a = 1e-4,
which these have not seen. 2000 steps on the free 5060 Ti: X-Trans ~2900
img/s, Bayer ~1850, so 500k steps are ~1.5 h / ~2.4 h; 24 MP `--gpu --bench`
f16: X-Trans 135 ms, Bayer 275. Export vs PyTorch (2000-step weights, with
the field through `run`): f32 86 / 84 dB, f16 57-58.

**CA in training, whole pixels** (`CA_WHOLE_PX`, `CA_SUPER` 1; for the 2026-10-18
models, which were told the field; off since 2026-10-23, removed with the
rest of the training CA 2026-10-25): the field is
rounded to whole pixels before red and blue are moved, so the move is a copy
of pixels (`shift` at an integer is exact), at the crop's own size, with no
blur; the network is told the rounded field, off by up to `CA_ERR` (10%).
It replaces the supersampling (`CA_SUPER` 2: sub-pixel shifts without blur),
which cost 6.4 ms a batch against 1.1 (0.8 with no CA). Of the pixels 58%
have no shift, ~27% 1 px, 11% 2 px, 3.5% 3 px or more. The network never sees
a sub-pixel field in training, and a real lens' is smooth: whether it carries
over is untested (the suite's "lateral CA" rows, which use a smooth sub-pixel
field, will say). If not: `CA_SUPER = 2`, `CA_WHOLE_PX = False`. Without
Catmull-Rom's ringing the CA batches no longer clamp the mosaic at 0.

The earlier checkpoints are `xtrans_v1.pt` and `bayer_v1.pt` (also in git;
their scripts are in git history before this change).

`Demosaic` (common.py): packed mosaic + pooled sigma -> trunk at 1/TILE
resolution -> `pixel_shuffle` to a full-res RGB estimate, plus a residual
full-res head (`HEAD` in `train_*.py`, `common.head`): on `[estimate,
scattered mosaic, CFA masks, noise std]`, a 3x3 with one weight bank per CFA
phase (`PhaseConv`, `HEAD_WIDTH` 32), a LeakyReLU and a 1x1 to RGB, added to
the estimate. Circular padding throughout. Input is the white-balanced mosaic and its noise std;
output is white-balanced camera RGB.

The trunk is residual: a stem 3x3, then pre-activation blocks
`x + conv3x3(LeakyReLU(x))` (`Residual`, zero-initialised so each starts as
identity), then LeakyReLU and a 3x3 to `3 * TILE^2`.

- Bayer (`train_bayer.py`): tile 2 (one period, `pixel_unshuffle`), width 32,
  14 blocks: 16 3x3s, reach 33 px, halo 34.
- X-Trans (`train_xtrans.py`): tile 3, width 64, 9 blocks: 11 3x3s, reach
  35 px, halo 36. A 3x3 cell of X-Trans comes in two kinds (each the other
  with R and B swapped, in a checkerboard), so the packing (`pack_layout`,
  `Demosaic.pack`; `PREP_SPLIT` in `wgsl.py`) splits the 4 non-green sites
  into R and B planes and adds a 0/1 plane for the cell's kind: 5 + 8 + 1 +
  sigma = 15 inputs. graph.txt's `multiple` is lcm(tile, period) = 6, so
  windows start on whole periods and the cell kind is the same in the window
  as in the image.

Reach is exactly `tile * (k3 + 1) - 1` (k3 = trunk 3x3s, each reaching one
cell; a pixel can sit at the far side of its own), measured by gradients for
all six networks without the head; the head's 3x3 adds 1 px, `tile * (k3 +
1)`, which every halo still covers. Each model passes its `HALO` (the reach rounded up to the
period) to `train` and `export`, which checks it against the reach and the
period; the loss drops `BORDER = HALO` px per side on 192 px crops.

In `wgsl.py` a residual layer's conv gets the LeakyReLU on its input load and
its own input added before the store, spliced into `CONV_TILED` / `CONV_1X1`
by `residual()`. (No current model has a 1x1 trunk layer; `CONV_1X1` stays as
a generic kernel.) Exports with random weights match PyTorch on test.bin at
f32 85-92 dB, f16 72-77 dB; the trained head models at f32 83-84, f16 66-67
(500k steps), the head-free xtrans at f16 58 dB (the same against the truth,
34.12 vs 34.13).
The f16 gap is the fp16 rounding of the weights, which both builds use and to
which a trained deep residual stack is sensitive: against PyTorch run on the
same rounded weights, with TF32 off, the f32 export matched at 140 dB
(X-Trans light). test.bin's reference is computed with TF32 convs (cuDNN's
default, and `train()` sets it), which on its own costs a few dB of that
comparison. 24 MP, `--gpu --bench`, f16 / f32 on the 5060 Ti, with the head:
X-Trans ~478 / 520 ms (~252 / 279 ms at width 48), Bayer ~378 / 512 ms; the
head-free X-Trans ~454 ms f16 (with the head, measured the same day: 469). The f16 tiled 3x3
kernel stages input channels in chunks of 16 (f32: 8), so in f16 a width
between 49 and 64 costs as much as 64. Training ran at ~790 (X-Trans, width
48) / ~650 (Bayer) img/s, so 500k steps take ~6-7 h each.

Profiled 2026-10-16 (head-free X-Trans, 5060 Ti). Training, batch 32, per
step: sensor (`random_batch`) 3.0 ms, forward 6.7, loss 2.8, backward 11.7,
optimizer and EMA 0.1: the GPU is never idle (the data loader alone gives
~25k img/s), so CUDA graphs cannot help; bf16 instead of fp16 + GradScaler
changed nothing, batch 64 was slower per image, max-autotune +1%. Computing
the tone error once for both loss terms and compiling the loss (`loss`,
`floss` in `train`) took the loss to 0.5 ms and backward to 9.3: 1319 ->
1640 img/s. Inference, 24 MP f16 / f32 450 / 484 ms: the nine 64 -> 64
residual 3x3s take 410 / 442 of it (~45 ms each, ~5 TFLOPS, about a fifth
of the GPU's ~24), everything else (cut, prep, stem, last conv, place) 40;
measured by timing the export with ops removed. `CONV_TILED` gave each
thread 4 px x 8 output channels (~5 FMAs per workgroup-memory load), and 8
workgroups staged the same input tile for the 64 output channels. It now
takes its shape as `TILED_ROWS` (output rows per invocation) and `TILED_OCB`
(output channels per workgroup, per layer width; `tiled_code` unrolls the
accumulators, loads and FMAs, `pack_tiled` groups the weights by it); the
output is bitwise the same for every shape (same terms, same order;
checked on test.bin f16 / f32 and the X-E1 through the app). Timed 4 x 8,
2-8 rows x 8-64 channels on every network, best of 3 runs at 24 MP: 3 rows
x 32 channels for 64-wide layers, 16 for narrower ones, was within ~10% of
the best everywhere: X-Trans 458 -> 307 ms f16, 520 -> 365 f32 (the X-E1
develop with readback ~515 -> 291 ms); Bayer 358 -> 262, 505 -> 343; lite
and tiny 16-27% faster in f16, 7-16% in f32 (random weights). Tried and
dropped: Winograd F(2x2, 3x3), fused like CONV_TILED (each invocation a 2x2
tile x 8 or 16 output channels, U = G g G^T packed at export, V = B^T d B
per input channel, Y = A^T M A at the end). Accurate (f16 59.6 dB vs
PyTorch, direct 58.8; f32 the same), but twice as slow (X-Trans f16 636 vs
303 ms): with 16 accumulators per channel group a thread holds one tile, so
each transformed weight is used once and each V for only 8-16 channels,
~2.7 FMAs per workgroup-memory load against the direct kernel's ~8.7. A
win would need Winograd as batched GEMMs (a thread on one of the 16
positions for a block of channels x tiles, M through workgroup memory to the
output transform), hard to fit in WebGPU's 16 KB. The old exports of the models still to be retrained keep the old
kernel until they are re-exported.

**Lateral CA as an input** (2026-10-16; removed from training 2026-10-23:
the softness it brought on some suite images could not be trained out, so
the CA is now taken out before the network, `CA_FIX`. `Demosaic`'s field
input, `common.widen`, `finetune.py`'s CA and head steps and the export's
`ca` buffer went with it): a model with `CA = True` in its
`train_*.py` also takes the lateral CA field: per pixel where red and blue
are against green (dy, dx each, px), four more planes after sigma
(`Demosaic`: `sigma` is then B,5,H,W; given only the noise std it is told no
CA), average-pooled per cell into the stem (X-Trans 19 inputs, Bayer 9). It
learns to undo it inside the demosaic, where the drawn correction (`lens/
measure.rs`) could only shift channels the network had already partly
pulled together (fine structures came out wrong). Training: see "Training
data". The export pools the field in the prep op from graph.txt's `ca`
buffer. `xtrans.pt` and `bayer.pt` take it (2026-10-17): each was
fine-tuned from the 500k-step model without it (`finetune.py`, which
widens the stem with zero weights for the new inputs, `common.widen`:
exactly the trained model; `CA_STEPS` 100k at `CA_LR` 5e-5, xtrans on
Modal, bayer locally) and promoted; judged better in every way in the app.
The models from before are kept as `xtrans_noca.pt` / `bayer_noca.pt`
(untracked). CA left in the output, as the drawn correction measures it
(rho 0.5 / 1): X-T30 X-Trans R -0.01 / -0.08, B -0.02 / +0.09 px (mosaic
+0.50 / +0.29, +1.11 / +2.94); A6700 Bayer R +0.05 / +0.06, B -0.05 / -0.17
(mosaic +0.30 / +0.35, -0.43 / -0.70); Z5 B +0.07 / +0.02 (mosaic +0.44 /
+0.68).

Until 2026-10-15 these were the "residual" alternatives (`*_res`), beside the
original models, which they replaced at the same quality (see "Tried and
dropped").

**The head, removed and restored** (2026-10-16 / 17): the head-free models
looked as good while the noise fit was high, but with an honest fit (or a =
b = 0) low-ISO images came out slightly soft at the pixel level: the head
was the only part reconstructing the finest detail (head-free xtrans: Kodak
38.31 vs 39.44, zone 29.2 vs 32.1). It came back as a residual on the
trained head-free models: `common.head`'s last 1x1 starts at zero, so a
model given a head is exactly the model without one, and `finetune.py`
trained it 100k steps at 1e-4. The head takes only sigma's noise plane, the trunk the CA field. The
export is the old folded kernel (`HEAD`, `fold_head`, `pack_head_out`)
plus the estimate added before the store; checked with a random head: f32
84-85 dB, f16 57-58 dB vs PyTorch (X-Trans, Bayer), and `--ca-check`
identical (f16 X-Trans) / 3e-3 (Bayer). The pixel-level CFA chroma leak the
head caused (see "Tried and dropped") is to be checked again on the result.

**State of the checkpoints (2026-10-16):** `xtrans.pt` (500k steps, from
scratch on Modal) and `bayer.pt` (500k, locally) are head-free and trained,
both with CA (fine-tuned 100k more, see above). bayer_lite, bayer_tiny, xtrans_lite and xtrans_tiny are
to be retrained; their head-model
checkpoints were deleted (they cannot load into the head-free networks), but
their old `*_wgpu/` exports are kept until then: wgpu-run runs any exported
graph, and release builds need `bayer_wgpu/`. Suite of the head-free xtrans
(head model in brackets): held-out 32.98 (33.40), Kodak 38.31 (39.44), zone
29.2 (32.1), pattern 77.2 (56.8).

### The light models

`train_xtrans_lite.py` / `train_bayer_lite.py` (-> `xtrans_lite.pt`,
`bayer_lite.pt`, `*_lite_wgpu/`): the models at half the width,
same depth, packing and reach, for weaker GPUs (the app's default on
an integrated GPU, see Debugging above). X-Trans width 32 (200k parameters),
Bayer width 16 (47k). 24 MP, `--gpu --bench`, f16 / f32 on the 5060 Ti, with
random weights: X-Trans ~129 / 139 ms, Bayer ~109 / 122 ms, so f32 (weak
GPUs often lack f16) costs them little (timed with the head). At this size
the head, packing and window cut and place were a large share: an X-Trans width-16 trunk still took
54 ms. Other shapes timed for the same reach: X-Trans tile 6 width 64 / 96
85 / 144 ms, Bayer tile 4 width 32 / 48 72 / 126 ms (tile 4 would need
graph.txt's `cfa` line tiled to 4x4 for `pad::phase`). Training: ~1100
(X-Trans) / ~870 (Bayer) img/s here. 200k steps (`STEPS`).

### The tiny models

`train_xtrans_tiny.py` / `train_bayer_tiny.py` (-> `xtrans_tiny.pt`,
`bayer_tiny.pt`, `*_tiny_wgpu/`): the light models at half the depth, for
the weakest GPUs. X-Trans 4 blocks: 6 3x3s, reach 20 px, halo 24 (a whole
period); Bayer 6 blocks: 8 3x3s, reach 17, halo 18. 100k steps. 24 MP,
`--gpu --bench`, f16 / f32 on the 5060 Ti, random weights, with the head: X-Trans ~72 / 81 ms, Bayer ~54 / 61 ms, about half of lite. The
app offers them in the picker (Tiny) when exported; `Engine::automatic`
still picks Light.

## Tried and dropped

**The original models** (removed 2026-10-15): the same packing-trunk-head
layout without residual blocks. X-Trans at tile 6 (one period, 36 planes),
three 3x3s at width 144 with 1x1s at 288 between them; Bayer at tile 2, nine
3x3s at width 48. Reach 24 / 20 px, halos 30 / 22 (`tile * (k3 + 1) + 1`
rounded to the tile), loss border 18. They scored about the same as the
residual models, which have more reach (36 / 34 px), train and export one
way, and could resume training (the originals' checkpoints, converted from an
older format, had no optimizer state). 24 MP f16 / f32: X-Trans ~342 / 471
ms (the residual one is slower in f16), Bayer ~387 / 547. f16 vs PyTorch on
test.bin: 62.0 / 67.4 dB. Timings and PSNRs below in "The wgpu export" that
predate the removal were measured on them.

**More trunk features into the head** (tried 2026-10-15): the trunk handing
the head E full-res channels instead of the 3 of the RGB estimate. As E more
outputs of the last 3x3 it cost (24 MP f16, E 3 -> 8) X-Trans +10 / +18 /
+31% (full / lite / tiny), Bayer +4 / +9 / +17%: the last conv's extra work is
9 * WIDTH * (E - 3) per px, and X-Trans models are twice as wide. Making the
last trunk conv a 1x1 instead (reach one cell less) brought E = 8 to today's
cost or 2-6% below it, but cannot be converted from a trained 3x3. The trial:
xtrans_lite with a 1x1 beside the last 3x3 (on its input) for the 5 new
features, head weights for them zero (an exact conversion), fine-tuned 30k
steps at LR 5e-5, against the unchanged model fine-tuned the same way. The
suite differed by -0.07..+0.09 dB per category, and test.bin vs the truth by
0.01 dB: no gain. (The fine-tune itself moved xtrans_lite more: edges +0.7,
clipped +0.6, lines +0.4, zone -1.0 dB; it was stopped at 240k steps of a
longer schedule, before its LR decayed. Not applied.)

**A loss relative to the signal level** (tried 2026-10-16): Charbonnier of
`(out - target) / (luma(target) + 1e-3)` in linear sRGB at the patch's
original exposure (so independent of how the image is later exposed; the
pattern loss on the same error), instead of the tone-mapped error. Fine-tuned
30k steps on copies of the X-Trans models: worse in every suite category,
-0.2..-1.2 dB (zone of lite and tiny +0.1..+0.7 the only gains), and on the
lifted synthetic shadows -0.05..-0.5 dB (xtrans lift 5: 26.17 -> 25.90);
visually the same as the originals in the app. Likely why: the loss's gamma
~0.45 is close to the square root that evens out shot noise, so it weights
each pixel by about what its signal-to-noise lets the network recover; the
relative loss weights the darkest pixels far beyond it. Dropped.

**The full-resolution head** (removed 2026-10-16, restored as a residual
2026-10-17, see "The model"): after the trunk's
pixel shuffle, a head on `[estimate, scattered mosaic, CFA masks, sigma]`,
a 3x3 with one weight bank per CFA phase (`PhaseConv`, 32 channels), a
LeakyReLU and a 1x1 to RGB; at export folded per phase to a 3x3 on
[estimate, m, sigma] (`HEAD`, `fold_head` in wgsl.py, ~10 ms per 24 MP).
It gave detail (head vs head-free xtrans, both 500k steps from scratch:
Kodak 39.44 vs 38.31, zone 32.1 vs 29.2), but it was where pixel-level
chroma noise in the CFA's pattern came from: each sample leaked into its
own channel. On flat grey with the X-E1's noise fit, the high-passed chroma
of R / G / B was 1.20-1.42 / 0.75-0.90 / 1.17-1.33x as strong at the
channel's own sites as elsewhere with the head, 0.93-1.04 without; the
fixed per-position pattern was far larger too (suite pattern 56.8 vs 77.2
dB). By eye the head-free model was nearly indistinguishable but for that
chroma noise, and it is simpler to train and export, so the head went. A
first head-free try fine-tuned the head model's trunk instead (its estimate
alone scored 11.3 dB: the head did far more than refine; 160k steps reached
held-out 32.02, zone 24.1) and showed artifacts training from scratch did
not.

**A full-resolution network:**

A "star" network (no packing and no head; every layer a per-phase conv,
blocks of four parallel 13-tap line convs, horizontal, vertical and both
diagonals, mixed by a 3x3; width 16, 4 blocks, reach 30 px, ~23k MACs/px,
plus a clipped-sample input) was trained for X-Trans (200k steps) and
removed. It scored between the light and the original models (held-out
32.01 dB vs the residual model's 32.71, Kodak 38.48 vs 39.67), close to it on colour lines
and edges but 2.7 dB behind it on dark noise; its WGSL export (f32
accumulation, 78 / 87 dB vs PyTorch) ran 24 MP in ~450 ms f16, no faster
than the residual model. Training it needed custom Triton kernels: cuDNN's dense form of
per-phase convs reached only 221 img/s.

`suite.py` scores trained models per category (held-out, Kodak from
`../demosaic/images/`, zone plate, line gratings, colour edges, clipping,
dark noise, symmetry under flips and rotations that map the CFA onto itself
after a shift; clean and noisy; *pattern*: flat grey at 0.5, 2 and 8% through
the sensor, 32 noise draws, -20 log10 of the sd of the output's mean error per
CFA position over the level; *push 3 / 5*: Kodak shot 3 / 5 EV under and
scored pushed back up, after the loss's display curve; *lift 3 / 5*: a
synthetic dark image (`shadows()`: float, -10..-6 EV, faint texture, colour
patches, faint lines and edges, bright discs) shot at its exposure and shown
3 / 5 EV brighter; each also as chroma alone; *CA fixed*: Kodak with
`TEST_CA` taken out of the mosaic by `ca_correct`, as the app does) and writes the outputs to
`training/suite/` (gitignored). `python suite.py` runs every `train_*.py`
whose checkpoint exists, or the ones named; a model with a `<name>_ft.pt`
also gets a "<name> ft" column.

## Checkpoints

`{'model': EMA weights, 'raw': trained weights, 'opt', 'scaler', 'step'}`.
Training resumes automatically from the file; inference and export read
`model`. Saved every 10000 steps, each save printing held-out loss and PSNR.

`xtrans.pt` and `bayer.pt` (the networks of 2026-10-23, see "The model")
start from scratch: the files there now are the 2026-10-18 plain models,
which must be moved aside first. The earlier ones are `*_v1.pt`, which no
script builds any more.

`finetune.py train_xtrans ..` trains a copy of a finished model with the
current loss and data (`FINETUNE_STEPS` 30k at `FINETUNE_LR` 2e-5, a fresh
optimizer and schedule from step 0) into `<name><FINETUNE_SUFFIX>.pt` and
`<name><FINETUNE_SUFFIX>_wgpu/` (gitignored), leaving the model alone; an
existing file of that name is resumed. The suffix is `_ftl` (2026-10-24), so
the 2026-10-17 `_ft.pt` models (the head ones the app showed as "WITH head")
are not touched; the app's picker lists `_ftl_wgpu` as "ftl" by itself and
`suite.py` gives it a column ("xtrans ftl"; the old `_ft.pt` files no longer load into the current architecture, so it skips them). The fine-tune adds
two terms to the loss (`common.loss(.., chroma, gradient)`, off = 0 = the
loss as it was, bit for bit): `chroma_loss`, Charbonnier of the tone-mapped
error's R - G and B - G (the error common to the channels cancels, what is
left is false colour and the CFA's chroma pattern), at `FINETUNE_CHROMA`
0.25, and `gradient_loss`, Charbonnier of the error's horizontal and vertical
differences (the error in edges and fine detail), at `FINETUNE_GRADIENT`
0.1. On a batch with the 2026-10-24 xtrans / bayer: Charbonnier 0.0131 /
0.0132, chroma 0.0159 / 0.0156, gradient 0.0157 / 0.0165, pattern 0.0011 /
0.0004 (x2 weighted), so the terms are ~30% and ~12% of the base loss. The
held-out loss printed at each save is still the plain Charbonnier, so it
compares with earlier runs. For a control (what the fine-tune does without
the terms, which moved scores by itself before) run it once with both 0 and
another `FINETUNE_SUFFIX`. Smoke run: 2000 steps of xtrans at ~1000 img/s. Result of the 30k-step
fine-tune (2026-10-25, suite, one run, no control without the terms), xtrans /
bayer, ftl minus base: held-out -0.01 / -0.01, Kodak 0.00 / 0.00, zone +0.05 /
+0.02, lines -0.01 / -0.04, edges -0.04 / +0.01, pattern -0.33 / -0.23, dark
+0.01 / -0.02, the chroma-only push / lift rows +0.10..+0.21 / +0.06..+0.15
(push 3 chroma 33.78 -> 33.92, push 5 chroma 28.98 -> 29.19, lift 5 chroma
27.01 -> 27.12), push / lift luma +0.04..+0.06 / 0.00..+0.05. To promote one, save it over the original with the original's
`step` (else `train_*.py` would train it on), then `--export`.

## Training data

DIV2K, crops cut online (`Crops`), the last 40 images held out. Each crop goes
through `random_batch` -> `sensor`: random camera matrix (8-camera bank blend +
hue/chroma jitter), R/B white-balance un-balance, a midtone shift
(`midtone_shift`, 2026-10-17, replaced the exposure of 1/8..2: mid-grey 0.18
lands at 1/8..2x itself, log-uniform, by a power curve on luma that leaves
black and white alone and the colour kept; the sensor's exposure is now 1, so
highlights clip no more than in the source and the draw keeps its stream),
clipping at SAT, mosaic, Poisson-Gaussian noise, a deliberately misestimated
sigma map, then re-balance. 20% of patches are noise-free; every 8th is
blurred; R/B are swapped with p=0.5. `CONTRAST_P` (25%) get more contrast
(`contrast_some`, 2026-10-17): log luma's
difference from its mean over the patch times 2-4, every channel scaled
alike, so the patch's geometric mean stays put and what goes past SAT clips.
It also strengthens DIV2K's own JPEG noise. Each pixel's factor is limited to
`CONTRAST_LIMIT` (16) either way: unbounded, a bright pixel in a dark patch got up
to 2.5e8x (7% of patches had a value above 100), the camera matrix made the other
channels hugely negative, and fp16 overflowed to a NaN loss every few thousand
steps (found 2026-10-17 with train_luma.py). Not yet trained with. (A local
version, detail above an 8x8-block base, was tried first: a hard 16x limit
on the change put bright halos along strong edges, a soft 2x one did not.) Loss: Charbonnier in tone-mapped linear
sRGB, `BORDER` px dropped per side. These were all verified to reproduce the earlier
code exactly at the same seed.

**Lateral CA** (removed 2026-10-25, see "How the app uses the models"; the
code is in git history before then): `ca_field` put the patch at a random
radius and angle in a frame of half-diagonal 2000-4500 px and gave red and
blue radial scale offsets with a corner shift uniform in ±4 px (0.4 px per
component at the median, 3.2 at the 99th percentile), `sensor` moved red
and blue by it before the mosaic (`shift`, Catmull-Rom: Keys' a = -0.75,
grid_sample's bicubic, moves small shifts 1.5x too far) and `ca_correct`
took it out with the field off by up to 10%. Shifting at the crop's own
size blurred red and blue (a sub-pixel Catmull-Rom shift keeps 69% of the
response at Nyquist at 0.25 px, none at 0.5), and models fine-tuned on that
(2026-10-17, 75% of patches) learned that red and blue have no fine detail:
X-Trans Kodak 38.31 -> 35.82 dB, zone 29.18 -> 19.90; Bayer Kodak 39.16 ->
36.20, zone 30.03 -> 16.68. Hence the supersampling at 2x (`CA_SUPER`,
every channel and the target averaged down after the shift), which did not
cure the softness either (see above). `sensor` still takes a field, for the
suite's "CA fixed" rows and `ca_fix_vector.py`, at the output size (so
their input red and blue are slightly soft).

**Hard-patch mining** (ported 2026-10-18 from the branch `hard-patch-mining`;
`MINE_*` / `MOIRE_*` in common.py, a model's `MINE` = the share of a batch it
replaces at the last step, 0 = off, which is the default: `train_xtrans.py` /
`train_bayer.py` have it at 0, `finetune.py` reads it; no model has been trained
with it): after `MINE_SCORE_START` (50%) of the steps, one batch in `MINE_EVERY`
(8) has its crops scored by `mine_scores`: through a fixed noise-free sensor
with no CA (camera matrix 0, fixed WB, no midtone shift or contrast) and the
network, the loss per patch and `moire_gain`, Gharbi et al. 2016's moire
metric (Lab Fourier gain of the output over the clean image below 0.95
Nyquist, blurred 3 bins, largest value per patch; their threshold 2, our
`MOIRE_ETA` guessed at 1: the paper does not give theirs). A crop whose loss
is over 1.5x the running mean of the scored batches' (`MINE_LOSS_RATIO`), or
whose gain is over `MOIRE_T`, goes into a ring of `MINE_BUFFER` (2048) crops,
kept as where it came from (image, y, x: `Crops(.., with_origin=True)`,
`Crops.cut`); from `MINE_START` (60%) a share of each batch rising linearly to
`MINE` at the last step is replaced by buffer crops, cut again at an offset of
up to `MINE_JITTER` (32 px of the patch, each way; real image context, not
padding, and a new place against the CFA) with a new rotation / flip, which go
through `random_batch` with fresh draws. Scoring is noise-free so it measures
the content's difficulty, not the draw's noise. The paper mines offline from
web photos with HDR-VDP2 (luminance) and this moire gain, on a first network's
output; we have 800 images, hence the replay. The buffer and the running mean
are not in the checkpoint: a resumed run mines from scratch. On the old trained
xtrans (1024 crops): loss over 1.5x mean 17.6%, moire over 2 1.7% (over 1:
12.6%), correlation of the two 0.22. Smoke-tested for 2000 steps on the new
Bayer (`MINE` 0.25: buffer 1107, 28% of the scored crops flagged, trains); its
quality effect is not measured.

**Pattern loss** (2026-10-16; measured on the head models): in dark flat areas the models left a fixed
pattern, a mean error per CFA position (sd ~1% of the level at 0.5% grey,
0.13% at 8%, X-Trans; constant in absolute terms), in X-Trans mostly at the
trunk's 3 px cell, which the drawn grain showed. Two additions train it out:
`flat_some` replaced `FLAT_P` = 10% of patches by their mean colour at a
log-uniform luma in `FLAT_LEVEL` with a random linear gradient up to
`FLAT_GRADIENT` (removed 2026-10-18, as it seemed to do little: only
`pattern_loss` is left, so the suite's `pattern` row is the one to watch;
the held-out set of a run from then on has no flat patches either, and the
random stream after the sensor draws moved), and `pattern_loss` (weight `PATTERN_WEIGHT` 2) adds the RMS
over CFA positions of each patch's mean tone-mapped error per position (mean
removed), so it costs nothing where the error averages out. The held-out
set includes the flat patches, so its PSNR is ~0.7 dB higher than before
for the same model. Applied by fine-tuning (`finetune.py`, 30k steps), suite
before -> after:

| | pattern | zone | kodak | edges | clipped | lines | dark |
|---|---|---|---|---|---|---|---|
| xtrans | 52.7 -> 56.8 | -0.7 | -0.2 | +0.6 | +0.4 | +0.4 | +0.1 |
| xtrans_lite | 50.3 -> 56.0 | -1.0 | -0.1 | +0.5 | +0.5 | +0.4 | +0.1 |
| xtrans_tiny | 38.4 -> 47.4 | -0.1 | -0.2 | -0.1 | 0.0 | 0.0 | -0.1 |
| bayer (not applied) | 56.8 -> 57.1 | -0.8 | -0.2 | +0.1 | +0.2 | +0.6 | +0.1 |

The zone and edge shifts in the full and lite models came from the
fine-tune itself (the head trial's control moved the same way without the
loss). The X-Trans models were promoted; Bayer's pattern was already small,
so its fine-tune (`bayer_ft.pt`) was kept out. bayer_lite (from 70k) and
bayer_tiny trained with it: pattern 56.9 / 58.9 (bayer_lite at 70k without
it: 43.7). Suite (2026-10-16), bayer / bayer_lite / bayer_tiny: held-out
33.38 / 31.34 / 28.83, Kodak 40.16 / 35.87 / 31.02, zone 34.6 / 21.7 / 14.3,
edges 26.4 / 25.2 / 22.0, dark 29.1 / 25.9 / 24.1 (xtrans_tiny: Kodak 34.4,
zone 17.7, edges 23.1).

## The wgpu export

Exports since 2026-10-23 have no CA input, and since 2026-10-25 wgpu-run
no longer runs the older ones that do (a `ca` line in graph.txt is a load
error): their `ca` buffer, `RawInput::ca`, `Model::takes_ca`, `run`'s field
argument, test.bin's field and `demosaic-run --ca-check` are gone (git
history before then has them).

`export()` writes `<name>_wgpu/{f16,f32}/` (graph.txt, one .wgsl per dispatch,
weights.bin) and `test.bin`. Everything model-specific -- kernel choice
(1x1 / tiled k x k / plain conv), template constants, weight
repacking, buffer reuse, dispatch sizes -- is decided in `wgsl.py`; the Rust
side only allocates and dispatches what graph.txt lists. Shaders are built for
one window (`WGPU_WINDOW` = 1056 px); `Model::run` processes any image as
windows with a halo that wrap around the image edge. f16 needs the adapter's
`SHADER_F16`.

**Zero padding** (2026-10-18; `common.conv`, `PhaseConv`, `wgsl.py`'s
`CONV` / `CONV_TILED` loads and `HEAD` taps): every conv pads with zeros, in
training, PyTorch inference and the shaders. It was circular, which in
PyTorch is a copy of every activation before every conv (~30% of a training
step; Bayer clean model fwd+bwd 42 -> 29.5 ms, X-Trans 29 -> 20.8). Nothing
counted sees the padding: the loss and the PSNR drop `BORDER` >= `HALO` >=
the reach per side (`evaluate` drops it from the PSNR too now, so its numbers
are higher than the earlier ones; a checkpoint scored 32.57 dB with the
border and circular, 30.75 with zeros, the same held-out loss), so the same
weights give the same loss and gradients, and circular and zero-padded
PyTorch agree on every pixel the loss counts (Bayer exactly, X-Trans 1e-5).
The shaders: a model's export differs from the circular one only in the
window's outer ring, which the halo keeps out of every core, so the
output is bitwise the same (Bayer and X-Trans clean models, `demosaic-run
--dump`, f16, test.bin sized), and old circular exports keep working
unchanged until they are re-exported. Checked directly with `halo 0` in a
copy of graph.txt and an image exactly one window big (no wrap-around in the
cut): every pixel but the outermost matches zero-padded PyTorch at the f16
level (max 5e-3 Bayer, 1e-2 X-Trans); the outermost ring differs because
`fold_head` puts the masks of taps beyond the edge in the bias, where PyTorch
has none. The cut shader still wraps the image round: the output within
`halo` px of the image's edge is therefore not PyTorch's (which sees zeros
there), and `demosaic-run` drops `halo` px (not 8) before its PSNR and max
error. The app pads by >= halo, so never outputs those pixels.

Weights are rounded to fp16 for both builds. The current models' accuracy
against PyTorch and their timings are in "The model". The original models'
trunk was verified bitwise identical to the earlier hand-built Rust runner
(whole image vs 6 windows, f16 and f32); the numbers below were measured on
those models.

Edge windows do only the work their core needs. A window whose core is cut
short by the image's end holds image data only in its first core + 2 * halo
rows and columns (`Valid`, a uniform at binding 5 that `Model` writes before
each window's submission), and the core reads nothing beyond them. Every
kernel scales that size to its own grid (`VALID`, `S` = window px per grid px)
and skips workgroups wholly outside it; the cut shader fills only that part.
Output is bitwise unchanged (all `--dump`s: test.bin and three raws, f16/f32,
`--gpu` or not), also with old shaders that ignore binding 5. Before, the
number of windows was `ceil(padded size / core)` per axis, so sizes just past a
multiple of the core (996 px X-Trans, 1012 Bayer) paid for nearly empty
windows: 15-30% of the work for most sensors after the app's padding. f16,
before -> after: X-Trans 6000x4000 458 -> 340 ms, 6048x4032 483 -> 352, 4896x3264
262 -> 222; Bayer 6240x4160 556 -> 410, 8256x5504 866 -> 722; sizes that tiled
well are unchanged (5976x3984: 316 either way). Throughput is now ~70 MP/s
(X-Trans) and ~63 MP/s (Bayer) whatever the size. The 1x1 kernels' 64-pixel
runs can straddle rows, so a run is skipped only when it starts beyond the
valid columns and ends on that row or on an invalid one.

The window was 1536 until 2026-10; 1056 (= 96 * 11, packed grid still a
multiple of the 16 px workgroups) gives bitwise identical output, keeps each
per-window submission ~2.2x shorter (Windows resets the GPU after ~2 s) and
halves the buffers (largest: Bayer f32 54 MB, was 113). On the 5060 Ti it was
faster anyway (X-Trans f16 681 -> 650 ms, Bayer f16 798 -> 500 ms, f32 874 ->
670; only X-Trans f32 slower, 815 -> 880), likely because the buffers now fit
the cache better. Keep the largest buffer under `max_storage_buffer_binding_size`
(128 MiB by default) if it is ever raised.

The graph's mosaic and sigma buffers are kept to the end (`write` marks
them as outliving every op): `run_texture`'s place shader reads them after
the graph for the grain. Nothing but the first op reads them, so the
output would otherwise take the mosaic's buffer (as in the first head-free
export: the grain, noise reduction below 100%, was garbage while D was
right). `write` also deletes the directory's old `.wgsl` files first, so an
export with fewer ops leaves none behind for build.rs to embed.

Any change to `wgsl.py` templates or packing: re-export and check
`demosaic-run` against test.bin (the PSNR vs PyTorch must not drop).
