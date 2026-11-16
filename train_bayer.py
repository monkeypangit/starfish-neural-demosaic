"""Train the residual Bayer demosaic + denoise network, then export it for wgpu.

a5cd8e3's network with a wider second half and a shared head, for every ISO
(common's full noise range): a residual trunk at tile
2, pre-activation blocks (common.Residual) at width 32, then at width 64
after one 3x3 that widens the stream, and DemosaicNet's full-resolution head
(common.head, shared by every CFA phase) whose output is the image.
train_bayer.py is the plain network with a residual head that preceded it.

    python train_bayer_res.py               # train (resumes bayer_res.pt if it exists)
    python train_bayer_res.py --export      # just export bayer_res.pt -> bayer_res_wgpu/
"""
import argparse

import torch.nn as nn

from common import BAYER, DATA, HERE, Demosaic, Residual, act, conv, export, head, icnr, train

CFA = BAYER
TILE = 2                     # pack 2x2 = one Bayer period into 4 planes at 1/2 res
CKPT = HERE / 'bayer_res.pt'
WIDTH_1 = 32                 # the residual stream's channels in the first half (8 per pixel) ...
LAYERS_1 = 9                 # ... 3x3s there: the stem and 8 blocks
WIDTH_2 = 64                 # ... and in the second (16 per pixel) ...
LAYERS_2 = 6                 # ... 3x3s there: the one that widens the stream, 4 blocks, the output conv
HEAD_WIDTH = 64              # the head's hidden channels
HEAD_PER_PHASE = False       # one head weight bank per CFA phase (4) instead of one shared
HEAD_RESIDUAL = False        # the head's output is the image (DemosaicNet), not added to the trunk's
HALO = 32                    # reach 2 * (15 + 1) - 1 = 31 px, 32 with the head's 3x3

# (a, b) of var = a*x + b, each log-uniform per patch; None: common's A_RANGE and B_RANGE (a 1e-5..1e-2,
# b 1e-8..1e-4), which the app's noise fit and sliders are clamped to. 20% of the patches have no noise
# at all (common.ZERO_NOISE_P)
NOISE = None

STEPS = 1_000_000
MINE = 0.0                   # hard-patch mining (common.MINE_*): the share of a batch it replaces at
                             # the last step, rising from 0 at MINE_START; 0 = off
BATCH = 32
PATCH = 192                  # crop size, a multiple of TILE
BORDER = HALO                # px per side the loss drops: the zero padding reaches no further
LR = 2e-4
EMA = 0.999
CLIP_GRAD = 5.0


def build():
    # Trunk at 1/2 resolution. Input: 4 packed mosaic planes + pooled sigma. The
    # stream is raw between the blocks (each is x + conv(LeakyReLU(x))), so
    # the LeakyReLU before the 3x3 that widens it and before the output conv
    # is theirs.
    net = nn.Sequential(
        conv(5, WIDTH_1, 3),
        *[Residual(WIDTH_1, 3) for _ in range(LAYERS_1 - 1)],
        act(), conv(WIDTH_1, WIDTH_2, 3),
        *[Residual(WIDTH_2, 3) for _ in range(LAYERS_2 - 2)],
        act(), conv(WIDTH_2, 3 * TILE * TILE, 3),     # -> pixel_shuffle(2): the full-res RGB estimate
    )
    icnr(net[-1], TILE)
    return Demosaic(CFA, TILE, net, head(CFA.shape[0], HEAD_WIDTH, HEAD_PER_PHASE, HEAD_RESIDUAL), HEAD_RESIDUAL)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA, help='prep.py output')
    ap.add_argument('--export', action='store_true', help=f'only export {CKPT.name}')
    args = ap.parse_args()
    if args.export:
        export(build, CFA, TILE, CKPT, args.data, HALO, NOISE)
    else:
        train(build, CFA, TILE, CKPT, STEPS, BATCH, PATCH, LR, EMA, CLIP_GRAD, args.data,
              BORDER, HALO, NOISE, MINE)
