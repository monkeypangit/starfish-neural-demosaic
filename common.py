"""Everything train_xtrans.py and train_bayer.py share: the CFA tables, the
network's building blocks, the synthetic sensor, the loss, the training loop,
the wgpu export and a minimal inference.

The two train_*.py files hold the model itself and its training constants; the
constants below define the *data*: how a clean photo is turned into the noisy
raw mosaic the network learns to undo. They are identical for both CFAs.
"""
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import wgsl

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------ data constants

DATA_DIR = HERE / 'data'     # where the user puts the training data (data/README.md)
DIV2K = DATA_DIR / 'DIV2K_train_HR'   # the DIV2K training PNGs, added by hand
DATA = DATA_DIR / 'div2k.npy'         # prep.py output: every DIV2K image, decoded, flat uint8
HOLDOUT_IMAGES = 40          # the last 40 images (DIV2K 0761-0800) are never trained on
EVAL_PATCHES = 256           # held-out crops scored at every checkpoint
EVAL_SEED = 1234
WORKERS = 4                  # DataLoader processes cutting crops

# Poisson-Gaussian noise, var = a*x + b in raw units, a and b log-uniform per patch
A_RANGE = (1e-5, 1e-2)
B_RANGE = (1e-8, 1e-4)
ZERO_NOISE_P = 0.2           # fraction of patches with no noise and an all-zero sigma map

# The sigma map is built from a, b multiplied by these (log-uniform) factors
# while the noise itself uses the true a, b: at inference sigma is estimated.
SIGMA_ERR_A_RANGE = (1 / 2, 2.0)
SIGMA_ERR_B_RANGE = (1 / 5, 5.0)

# White-balance gains (raw -> balanced) for R and B; G is the 1.0 anchor
WB_R_RANGE = (1.3, 2.8)
WB_B_RANGE = (1.2, 2.6)

MIDTONE_RANGE = (1 / 8, 2.0)  # where mid-grey (MIDTONE_PIVOT) lands, as a multiple of itself,
MIDTONE_PIVOT = 0.18         # log-uniform: -3 .. +1 EV there. Not an exposure (the sensor's gain
                             # stays 1): a power curve on luma that leaves black and white alone,
                             # so highlights clip no more than in the source
SAT = 1.0                    # sensor white level in raw units

BLUR_EVERY = 8               # every 8th patch of a batch gets a Gaussian blur ...
BLUR_MAX_SIGMA = 2.0         # ... with sigma uniform in (0, 2] px
RB_SWAP_P = 0.5              # probability of swapping R and B of the source patch
BOOST_P = 0.25               # fraction of patches whose source gets more saturated, so the
BOOST_RANGE = (1.5, 3.0)     # deep colours DIV2K lacks appear: chroma x uniform in this range

CONTRAST_P = 0.25            # fraction of patches whose source gets more contrast, for the high-
CONTRAST_GAIN = (2.0, 4.0)   # contrast scenes DIV2K has few of: log luma about the patch's own
                             # mean log luma (its geometric mean stays put) x uniform in this range
CONTRAST_LIMIT = 16.0        # no pixel is scaled by more than this, or by less than 1/this (4 stops)

# Lateral chromatic aberration: the app takes it out of the mosaic before the
# network (src/processing/ca_fix.rs, which is ca_correct, checked against it by
# ca_fix_vector.py). Training has no CA; the suite's "CA fixed" rows put it in
# and take it out with ca_correct, as the app does. ca_correct's settings:
CA_FIX_RADIUS = 2            # ca_correct: the guided filter's window radius, px ...
CA_FIX_EPS = 1e-3            # ... its regulariser (balanced units squared), towards the colour ratio ...
CA_FIX_SIGMA = 0.5           # ... the Gaussian, px, that weights samples by how near they land ...
CA_FIX_GATHER = 3            # ... px around each site's rounded shift the samples are looked for in ...
CA_FIX_REACH = 8             # ... the largest rounded shift, px (more is clamped) ...
CA_FIX_NEAR = 0.01           # ... the weight from which a sample bounds the result (clamp)
CA_FIX_PRIOR = 'ratio'       # a's prior: the window's mean c / mean G ('zero': the guided filter's) ...
CA_FIX_RATIO_MAX = 4.0       # ... bounded to 0..this: where a saturated colour has almost no green the
                             # ratio exploded (kodim05 with a sharp green: errors up to 16x white)
CA_FIX_CLAMP = False         # clamp: on, Kodak through the models without CA lost 0.4-0.8 dB

HUE_MAX_DEG = 6.0            # camera-matrix jitter: hue rotation about the neutral axis
SAT_RANGE = (0.85, 1.20)     # camera-matrix jitter: chroma scale

# Loss: Charbonnier in tone-mapped linear sRGB, border dropped (each model's BORDER)
LOSS_EPS = 1e-6
# Plus PATTERN_WEIGHT x the RMS of the error's mean per CFA position (pattern_loss):
# a fixed error at each position of the CFA period, which the per-pixel loss barely
# sees under the noise but the eye sees on smooth dark areas
PATTERN_WEIGHT = 2.0
# Hard-patch mining (train, a model's MINE; 0 = off): from MINE_SCORE_START of the steps on, one batch in MINE_EVERY
# has its crops scored through a fixed noise-free sensor (`mine_scores`) and the ones that
# come out hard are noted in a buffer of MINE_BUFFER crops: a loss over MINE_LOSS_RATIO x
# the running mean of the scored batches' losses, or colour moire (`moire_gain`, after
# Gharbi et al. 2016: the demosaiced image's Fourier gain over the clean one's, in Lab,
# above MOIRE_T). From MINE_START on, a share of every batch's crops, rising linearly
# from 0 to MINE at the last step, is replaced by buffer crops: the buffer holds
# where each came from (image, y, x), and a replay cuts the image again at an offset of up
# to MINE_JITTER px (of the patch, in each direction) and a new rotation / flip, so the
# hard structure sits somewhere else against the CFA and the crop's edges, and goes through
# `random_batch` again with fresh noise and midtone shift.
MINE_START = 0.6
MINE_SCORE_START = 0.5
MINE_EVERY = 8
MINE_BUFFER = 2048
MINE_JITTER = 32
MINE_LOSS_RATIO = 1.5
MINE_WB = (1.9, 1.8)         # the scoring sensor's R and B gains (about the training ranges' middle)
MOIRE_R = 0.95               # frequencies below this fraction of Nyquist are compared ...
MOIRE_SIGMA = 3.0            # ... the gain map blurred by this many frequency bins ...
MOIRE_T = 2.0                # ... and a patch is aliased when its largest gain exceeds this
MOIRE_ETA = 1.0              # added to both energies (Lab^2 per pixel, orthonormal FFT)
WARMUP = 500                 # steps of linear LR warm-up before the cosine decay
WEIGHT_DECAY = 1e-4

# The test vector written next to an export: a held-out image through a fixed sensor
TEST_IMAGE = -HOLDOUT_IMAGES  # first held-out image
TEST_SIZE = (768, 1152)      # h, w; multiples of 6
TEST_WB = (2.0, 1.6)
TEST_A, TEST_B = 1e-3, 1e-6
TEST_CA = (2.0, -2.5)        # suite.py's and ca_fix_vector.py's CA: red's and blue's shift at the corner, px

# The wgpu build is compiled for windows of this many px a side (a multiple of
# every TILE); images of any size are processed as overlapping windows, one GPU
# submission each. 1056 = 96 * 11 keeps the packed grid a multiple of the 16 px
# workgroups for both TILEs. Smaller than the earlier 1536 so one submission
# stays short on slow GPUs (Windows resets the GPU after ~2 s) and the buffers
# small, at a few percent more halo work.
WGPU_WINDOW = 1056

# ------------------------------------------------------------------ the CFAs
# 0 = R, 1 = G, 2 = B

BAYER = torch.tensor([[0, 1],
                      [1, 2]])

# dcraw/LibRaw's X-Trans phase
XTRANS = torch.tensor([[1, 1, 0, 1, 1, 2],
                       [1, 1, 2, 1, 1, 0],
                       [2, 0, 1, 0, 2, 1],
                       [1, 1, 2, 1, 1, 0],
                       [1, 1, 0, 1, 1, 2],
                       [0, 2, 1, 2, 0, 1]])


def cfa_masks(cfa, h, w):
    """1,3,h,w one-hot colour of every site; h, w multiples of the CFA period."""
    p = cfa.shape[0]
    onehot = F.one_hot(cfa.long(), 3).permute(2, 0, 1).float()     # 3,p,p
    return onehot.repeat(1, h // p, w // p)[None]


# ------------------------------------------------------------- network parts

def conv(i, o, k):
    """k x k conv, zero padded. The models see only the reach around a pixel,
    so what pads the image reaches no pixel further from its edge than that:
    the loss drops that border (each model's BORDER), and the app and the
    wgpu windows (halo) keep every output pixel that far from it. (It was
    circular until 2026-10-18; a copy of every activation before every conv
    cost ~30% of a training step.)"""
    return nn.Conv2d(i, o, k, padding=k // 2)


def act():
    return nn.LeakyReLU(0.1, inplace=True)


class Residual(nn.Module):
    """x + conv(LeakyReLU(x)), pre-activation, so the stream x passes every
    block untouched. The conv starts at zero: each block starts as identity,
    which a deep stack without normalisation needs to train stably."""

    def __init__(self, c, k):
        super().__init__()
        self.conv = conv(c, c, k)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return x + self.conv(F.leaky_relu(x, 0.1))


def icnr(c, tile):
    """Init a conv that feeds pixel_shuffle(tile) so all tile^2 output slots of a
    channel start identical: no checkerboard on the tile grid at step 0."""
    t2 = tile * tile
    o, i, kh, kw = c.weight.shape
    sub = torch.empty(o // t2, i, kh, kw)
    nn.init.kaiming_uniform_(sub, a=5 ** 0.5)
    bias = torch.empty(o // t2).uniform_(-(i * kh * kw) ** -0.5, (i * kh * kw) ** -0.5)
    with torch.no_grad():
        c.weight.copy_(sub.repeat_interleave(t2, dim=0))
        c.bias.copy_(bias.repeat_interleave(t2))


class PhaseConv(nn.Module):
    """A k x k conv with one weight bank per CFA phase, each bank applied only
    at its own pixels. Phase of pixel (y, x) is (y % per) * per + x % per.

    Weight layout is bank-major, (per^2 * o, i, k, k), which is what the wgpu
    runner reads. All banks start identical.
    """

    def __init__(self, i, o, k, per):
        super().__init__()
        self.per, self.o = per, o
        self.conv = nn.Conv2d(i, o * per * per, k)
        with torch.no_grad():
            self.conv.weight.copy_(self.conv.weight[:o].repeat(per * per, 1, 1, 1))
            self.conv.bias.copy_(self.conv.bias[:o].repeat(per * per))

    def forward(self, x):
        p, n, o = self.per, self.per * self.per, self.o
        k = self.conv.kernel_size[0]
        B, C, H, W = x.shape
        xp = F.pad(x, (k // 2,) * 4)
        # im2col: every pixel's k x k neighbourhood stacked into channels, in
        # the order of weight[:, ci, dy, dx]
        cols = torch.stack([xp[..., dy:dy + H, dx:dx + W]
                            for dy in range(k) for dx in range(k)], 2).flatten(1, 2)
        # sort pixels by phase: B, C*k*k, n, H/p, W/p
        cols = F.pixel_unshuffle(cols, p).unflatten(1, (C * k * k, n))
        w = self.conv.weight.reshape(n, o, C * k * k)
        y = torch.einsum('bkqhw,qok->boqhw', cols, w)
        y = y + self.conv.bias.view(n, o).t()[None, :, :, None, None]
        return F.pixel_shuffle(y.flatten(1, 2), p)                  # B, o, H, W


def head(period, width, per_phase=False, residual=True):
    """The full-resolution head on [estimate, scattered mosaic, CFA masks,
    sigma] (10 channels): a 3x3 shared by every pixel, which finds the CFA
    phase from the masks and the scattered samples around it (`per_phase`:
    one weight bank per phase instead, PhaseConv), a LeakyReLU and a 1x1 to
    RGB. `residual`: that 1x1 starts at zero, so the model starts as its
    trunk alone, and Demosaic adds the head to the estimate; otherwise the
    head's output is the image, as in DemosaicNet (Gharbi et al. 2016: a 3x3
    to 64, ReLU, a 1x1, on the mosaic and the upsampled trunk; this one also
    sees the masks and sigma). The trunk ends in a linear conv shared by
    every position of its cell; the head is a small nonlinear network at
    every pixel, on the estimate and the samples around it."""
    out = conv(width, 3, 1)
    if residual:
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
    first = PhaseConv(10, width, 3, per=period) if per_phase else conv(10, width, 3)
    return nn.Sequential(first, act(), out)


def pack_layout(cfa, tile):
    """How the trunk's input is packed when `tile` is not a multiple of the CFA
    period, so a site of the tile x tile cell does not always have one colour.

    Cells repeat every r x r (r = lcm(period, tile) / tile); cell phase
    q = (cy % r) * r + cx % r. Returns (planes, types, r):

        planes  [(site, colour or None, mask)]: in site order, one plane per
                site if its colour never changes (colour None), else one per
                colour it takes, holding the sample only where it has that
                colour; mask = bit q set where the plane holds the sample
        types   [mask]: one 0/1 plane per kind of cell beyond the first (cells
                differ in their colour pattern), set where the cell is that kind

    Where tile is a multiple of the period this is plain pixel_unshuffle: one
    plane per site, no types. X-Trans at tile 3 has two kinds of cell (each the
    other with R and B swapped), so its 4 non-green sites split in two: 5 green
    + 8 R/B planes + 1 type plane.
    """
    p = cfa.shape[0]
    r = math.lcm(p, tile) // tile
    cfa = [[int(c) for c in row] for row in cfa]
    pattern = [tuple(cfa[((q // r) * tile + s // tile) % p][((q % r) * tile + s % tile) % p]
                     for s in range(tile * tile)) for q in range(r * r)]
    planes = []
    for s in range(tile * tile):
        colours = sorted({pat[s] for pat in pattern})
        for c in colours:
            mask = sum(1 << q for q in range(r * r) if pattern[q][s] == c)
            planes.append((s, None if len(colours) == 1 else c, mask))
    kinds = list(dict.fromkeys(pattern))
    types = [sum(1 << q for q in range(r * r) if pattern[q] == k) for k in kinds[1:]]
    return planes, types, r


class Demosaic(nn.Module):
    """pack the mosaic into tile^2 planes at 1/tile resolution -> trunk `net`
    -> pixel_shuffle back to a full-resolution RGB estimate, to which `head`
    (if any, see `head()`) adds what it finds at full resolution from the
    estimate, the measured samples, the CFA masks and the noise std (or, with
    head_residual False, which it replaces by what it makes of them). If tile
    is not a multiple of the CFA period the packing is pack_layout's.

    m, sigma: B,1,H,W white-balanced mosaic and its noise std. -> B,3,H,W in
    white-balanced camera space. The trunk sees sigma average-pooled per cell.
    """

    def __init__(self, cfa, tile, net, head=None, head_residual=True):
        super().__init__()
        self.tile = tile
        self.net = net
        self.head = head
        self.head_residual = head_residual
        self.register_buffer('cfa', cfa, persistent=False)
        self.layout = pack_layout(cfa, tile)

    def pack(self, m):
        """pack_layout's planes and type planes of m: B,1,H,W -> B,n,H/t,W/t."""
        t = self.tile
        planes, types, r = self.layout
        cells = F.pixel_unshuffle(m, t)
        h, w = cells.shape[-2:]
        q = (torch.arange(h, device=m.device) % r)[:, None] * r + torch.arange(w, device=m.device) % r
        masks = torch.tensor([mask for _, _, mask in planes] + types, device=m.device)
        on = ((masks[:, None, None] >> q) & 1).to(m)             # n, h, w
        sites = torch.tensor([s for s, _, _ in planes], device=m.device)
        return torch.cat([cells[:, sites] * on[:len(planes)],
                          on[len(planes):].expand(m.shape[0], -1, -1, -1)], 1)

    def forward(self, m, sigma):
        t = self.tile
        if self.layout[2] == 1:
            x = torch.cat([F.pixel_unshuffle(m, t), F.avg_pool2d(sigma, t)], 1)
        else:
            x = torch.cat([self.pack(m), F.avg_pool2d(sigma, t)], 1)
        estimate = F.pixel_shuffle(self.net(x), t)                  # B,3,H,W
        if self.head is None:
            return estimate
        masks = cfa_masks(self.cfa, m.shape[-2], m.shape[-1]).to(m)
        scattered = m * masks                    # each sample in its own colour plane
        out = self.head(torch.cat(
            [estimate, scattered, masks.expand(m.shape[0], -1, -1, -1), sigma], 1))
        return estimate + out if self.head_residual else out


# -------------------------------------------------------- the synthetic sensor

def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def apply_ccm(x, m):
    """x: B,3,H,W; m: B,3,3 -> B,3,H,W."""
    return torch.einsum('bij,bjhw->bihw', m.to(x), x)


# DNG ColorMatrix (XYZ D65 -> camera) of eight cameras; training draws a random
# blend of them per patch so the network does not learn one camera's colours.
SRGB_TO_XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
CAM_XYZ_BANK = torch.tensor([
    [[1.0405, -0.3755, -0.1270], [-0.5461, 1.3787, 0.1793], [-0.1040, 0.2015, 0.6785]],  # Nikon D850
    [[0.5488, -0.0909, -0.0437], [-0.5769, 1.3284, 0.2732], [-0.1041, 0.2135, 0.8524]],  # Nikon D700
    [[0.6722, -0.0635, -0.0963], [-0.4287, 1.2460, 0.2028], [-0.0908, 0.2162, 0.5668]],  # Canon 5D III
    [[0.5271, -0.0712, -0.0347], [-0.6153, 1.3653, 0.2763], [-0.1601, 0.2366, 0.7242]],  # Sony A7
    [[0.7610, -0.2124, -0.0512], [-0.4971, 1.2600, 0.2560], [-0.0776, 0.1503, 0.6015]],  # Panasonic GH5
    [[0.7368, -0.2260, -0.0587], [-0.3745, 1.1455, 0.2323], [-0.0528, 0.1345, 0.6994]],  # Fujifilm X-T3
    [[0.8756, -0.3050, -0.0902], [-0.4732, 1.2678, 0.2338], [-0.0902, 0.1140, 0.7042]],  # Nikon Z7
    [[0.7178, -0.1584, -0.1010], [-0.4519, 1.2557, 0.2247], [-0.0844, 0.1755, 0.6301]],  # Canon R5
])
# linear sRGB -> camera, rows normalised to 1 so neutral stays neutral
CAM_FROM_SRGB_BANK = CAM_XYZ_BANK @ SRGB_TO_XYZ
CAM_FROM_SRGB_BANK = CAM_FROM_SRGB_BANK / CAM_FROM_SRGB_BANK.sum(2, keepdim=True)


def loguniform(n, lo, hi, dev):
    return (torch.rand(n, 1, 1, 1, device=dev) * math.log(hi / lo) + math.log(lo)).exp()


def random_ccm(n, dev):
    """(cam_from_srgb, srgb_from_cam), each n,3,3: a sharpened random blend of
    the bank, then a small hue rotation and chroma scale (both keep neutral)."""
    bank = CAM_FROM_SRGB_BANK.to(dev)
    w = torch.rand(n, len(bank), device=dev) ** 3
    M = torch.einsum('nk,kij->nij', w / w.sum(1, keepdim=True), bank)
    s = (torch.rand(n, device=dev) * (SAT_RANGE[1] - SAT_RANGE[0]) + SAT_RANGE[0])[:, None, None]
    chroma = s * torch.eye(3, device=dev) + (1 - s) * torch.full((3, 3), 1 / 3, device=dev)
    # rotation about the neutral axis (Rodrigues)
    K = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]], device=dev) / math.sqrt(3)
    th = ((torch.rand(n, device=dev) * 2 - 1) * math.radians(HUE_MAX_DEG))[:, None, None]
    hue = torch.eye(3, device=dev) + th.sin() * K + (1 - th.cos()) * (K @ K)
    M = chroma @ hue @ M
    M = M / M.sum(2, keepdim=True)
    return M, torch.linalg.inv_ex(M)[0]


def sensor(lin, cfa, cam_from_srgb, srgb_from_cam, gain, e, a, b, sa, sb, ca=None):
    """Photograph a linear-sRGB image with a simulated raw sensor.

    lin: B,3,H,W linear sRGB. cam_from_srgb / srgb_from_cam: B,3,3. gain: B,3,1,1
    WB gains (raw -> balanced). e: exposure; a, b: true noise; sa, sb: the noise
    coefficients the sigma map is built from. All B,1,1,1. ca: B,4,H,W lateral
    CA (as radial_ca) that moves the mosaic's red and blue, not the target's
    (the suite's and ca_fix_vector.py's; training has none).

    camera matrix -> exposure -> un-balance to raw -> (CA) -> clip at SAT -> mosaic -> noise -> clip -> sigma -> balance mosaic and
    sigma per site.

    Returns (noisy B,1,H,W, sigma B,1,H,W, target B,3,H,W, srgb_from_cam B,3,3):
    noisy, sigma and target in white-balanced camera space; srgb_from_cam takes
    that to linear sRGB at the original exposure.
    """
    raw = apply_ccm(lin, cam_from_srgb) * e / gain
    seen = raw
    if ca is not None:
        seen = torch.cat([shift(raw[:, :1], ca[:, :2]), raw[:, 1:2], shift(raw[:, 2:], ca[:, 2:])], 1)
    raw = raw.clamp(max=SAT)
    # Catmull-Rom rings a little at edges; without CA, seen is raw (kept below 0, as the noise needs)
    seen = seen.clamp(0, SAT) if ca is not None else raw
    masks = cfa_masks(cfa, raw.shape[-2], raw.shape[-1]).to(lin)
    m = (seen * masks).sum(1, keepdim=True)
    noisy = (m + torch.randn_like(m) * (a * m.clamp(min=0) + b).sqrt()).clamp(max=SAT)
    sigma = (sa * noisy.clamp(min=0) + sb).sqrt()
    site_gain = (gain * masks).sum(1, keepdim=True)
    # target: balanced, clamped at the common white level so blown highlights are neutral
    target = (raw * gain).clamp(max=SAT)
    return noisy * site_gain, sigma * site_gain, target, srgb_from_cam / e.view(-1, 1, 1)


LUMA_SRGB = torch.tensor([0.2126, 0.7152, 0.0722])


def random_batch(clean_u8, cfa, info=None, noise=None):
    """B,3,H,W uint8 sRGB crops (on the GPU) -> a training batch from `sensor`
    with every nuisance drawn at random per patch. If `info` is a dict it is
    filled with the draws (swap, boost, e, midtone, a, b, sa, sb, gain, contrast), for the
    preview; the batch itself is unchanged. `noise`: (A_RANGE, B_RANGE) of
    a model trained on other noise than the defaults."""
    a_range, b_range = noise or (A_RANGE, B_RANGE)
    dev, n = clean_u8.device, clean_u8.shape[0]
    lin = srgb_to_linear(clean_u8.float() / 255)
    swap = torch.rand(n, 1, 1, 1, device=dev) < RB_SWAP_P
    lin = blur_every(torch.where(swap, lin.flip(1), lin))
    cam_from_srgb, srgb_from_cam = random_ccm(n, dev)
    gr = loguniform(n, *WB_R_RANGE, dev)
    gain = torch.cat([gr, torch.ones_like(gr), loguniform(n, *WB_B_RANGE, dev)], 1)
    shift_to = loguniform(n, *MIDTONE_RANGE, dev)     # drawn here to keep the random stream
    noisy = (torch.rand(n, 1, 1, 1, device=dev) >= ZERO_NOISE_P).float()
    a = loguniform(n, *a_range, dev) * noisy
    b = loguniform(n, *b_range, dev) * noisy
    sa = a * loguniform(n, *SIGMA_ERR_A_RANGE, dev)
    sb = b * loguniform(n, *SIGMA_ERR_B_RANGE, dev)
    # drawn last so the draws above keep their random stream; applied to the source
    boost = torch.where(torch.rand(n, 1, 1, 1, device=dev) < BOOST_P,
                        torch.rand(n, 1, 1, 1, device=dev) * (BOOST_RANGE[1] - BOOST_RANGE[0]) + BOOST_RANGE[0],
                        torch.ones(n, 1, 1, 1, device=dev))
    luma = (lin * LUMA_SRGB.to(dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    lin = (luma + boost * (lin - luma)).clamp(min=0)  # chroma about the luma, kept in gamut's positive octant
    lin, contrast = contrast_some(lin)
    lin = midtone_shift(lin, shift_to)
    e = torch.ones_like(shift_to)                     # the sensor's exposure: none
    if info is not None:
        info.update(swap=swap, e=e, midtone=shift_to, a=a, b=b, sa=sa, sb=sb, gain=gain,
                    boost=boost, contrast=contrast)
    return sensor(lin, cfa, cam_from_srgb, srgb_from_cam, gain, e, a, b, sa, sb)


def shift(x, d):
    """x: B,1,H,W; d: B,2,H,W (dy, dx) px -> x at (p - d(p)) for every p, so what
    is at p moves to p + d: Catmull-Rom, the edges repeated. It must be
    Catmull-Rom (or better): Keys' cubic at a = -0.75 (grid_sample's bicubic)
    moves small shifts 1.5x as far, which would teach the network wrong
    distances."""
    b, _, h, w = x.shape
    y = torch.arange(h, device=x.device).view(1, h, 1) - d[:, 0]
    z = torch.arange(w, device=x.device).view(1, 1, w) - d[:, 1]
    iy, iz = y.floor(), z.floor()

    def weights(f):
        return [f * (-0.5 + f * (1 - 0.5 * f)), 1 + f * f * (-2.5 + 1.5 * f),
                f * (0.5 + f * (2 - 1.5 * f)), f * f * (-0.5 + 0.5 * f)]
    wy, wz = weights(y - iy), weights(z - iz)
    flat = x.reshape(b, h * w)
    out = torch.zeros_like(y)
    for i in range(4):
        row = (iy + i - 1).clamp(0, h - 1).long() * w
        for j in range(4):
            col = (iz + j - 1).clamp(0, w - 1).long()
            out = out + wy[i] * wz[j] * flat.gather(1, (row + col).reshape(b, -1)).reshape(b, h, w)
    return out[:, None]


def ca_correct(m, cfa, field, clamp=CA_FIX_CLAMP, green=None):
    """The lateral CA taken out of a mosaic, before the demosaic: m B,1,H,W
    (balanced), field B,4,H,W: red's (dy, dx), then blue's, against green, px
    (as radial_ca; the red sample at site q saw the scene at q - d(q)) -> the mosaic with red and blue as if the lens had none;
    green is left alone. Any CFA.

    Per colour c, a guided filter with a moved guide: G is green interpolated
    (normalised 3x3 tent over the green sites), Gs = G at each site's scene
    point (shift); over the c sites of each window, c = a Gs + b by least
    squares (CA_FIX_RADIUS), a box-averaged as in a guided filter (b drops
    out: it is the same at q as at p). CA_FIX_EPS regularises a towards the
    colour ratio mean c / mean Gs (CA_FIX_PRIOR), not towards 0 as a guided
    filter does: with a field that is near constant locally, a sample's
    move onto p comes from a alone, and on real raws fine texture and noise
    shrank a so that a third of the CA was left (Z5 blue 0.16 of 0.48 px at
    rho 0.55, measured in the corrected mosaic); with the ratio prior
    <= 0.02 px red, <= 0.09 blue on the Z5 and an X-T3, and the same on
    Kodak. The output at a c site p is the c samples whose scene points lie
    nearest p, each moved by the model, c(q) + a (G(p) - Gs(q)): Gaussian
    weights of CA_FIX_SIGMA px on how far q - d lands from p, normalised.
    With a = 0 that is plain interpolation of the samples; the model moves
    them the fraction of a site to p along green's structure. `clamp` keeps the result within the samples it came
    from (no overshoot, no new clipping). `green`: a full-res green to use
    instead of the interpolated one (the true one, for a bound).

    G is the tent at every site, green ones too (until 2026-10-24 green
    sites kept their sample): the move adds a (G(p) - Gs(q)), which on flat
    areas is only green's noise, and a guide of raw samples at half its
    sites passed it on at full strength. Bayer, ISO 800, shifts of 0.5-1.5
    px: the green noise in red and blue 0.38-0.44 -> 0.27-0.33 of their own
    noise's sd, the error without noise 0.1-0.35 dB more. (RCD's green, on a
    first corrected mosaic, was +0.4 dB but put 0.5-1.0 in.)

    G limits it: the networks' green is far better (41-44 dB on Kodak, from
    a CA-free mosaic) but takes it from red and blue, so with CA in them it
    falls to ~31 and a correction built on it is worse than on the tent;
    green from green samples alone is ~28-30 dB at the other sites (a
    gradient-weighted directional one was no better than the tent on X-Trans,
    +0.5 dB on Bayer). Kodak, TEST_CA, through the models without CA: X-Trans
    37.3 dB without CA, 29.5 with it, 32.8 corrected, 37.0 with the true G;
    Bayer 39.9 / 28.8 / 33.2 / 39.1. The fringes are gone by eye."""
    b, _, h, w = m.shape
    masks = cfa_masks(cfa, h, w).to(m)
    tent = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=m.device).view(1, 1, 3, 3)
    mr, mg, mb = masks[:, :1], masks[:, 1:2], masks[:, 2:]
    g = green if green is not None else \
        F.conv2d(m * mg, tent, padding=1) / F.conv2d(mg, tent, padding=1).clamp(min=1e-6)
    k = 2 * CA_FIX_RADIUS + 1
    box = lambda x: F.avg_pool2d(x, k, 1, CA_FIX_RADIUS, count_include_pad=False)
    # red and blue sites never share a pixel: each site's own colour's field, slope, etc.
    d = mr * field[:, :2] + mb * field[:, 2:]
    gs = shift(g, d)
    a = torch.zeros_like(m)
    for mc in (mr, mb):
        mc = mc.expand_as(m)
        n = box(mc).clamp(min=1e-6)
        mean_c, mean_g = box(mc * m) / n, box(mc * gs) / n
        var = box(mc * gs * gs) / n - mean_g ** 2
        a0 = (mean_c / mean_g.clamp(min=1e-4)).clamp(0, CA_FIX_RATIO_MAX) if CA_FIX_PRIOR == 'ratio' else 0.0
        a = a + mc * box((box(mc * m * gs) / n - mean_c * mean_g + CA_FIX_EPS * a0) / (var + CA_FIX_EPS))
    # gather, at each red or blue site p, the samples q of its colour in the 7x7 around
    # p + round(d(p)) (d is smooth over a few px; the Gaussian is ~0 beyond): sum w (c(q) -
    # a Gs(q)) and sum w a, so the result is that plus (sum w a) G(p). A fixed number of
    # taps, as the app's shader, so it compiles (random_batch); flattened with a border so
    # every tap is one offset and one gather
    colour = (mg + 2 * mb).expand_as(m)                     # 0 R, 1 G, 2 B
    r = CA_FIX_GATHER
    pad = CA_FIX_REACH + r
    ry, rx = d[:, :1].round().clamp(-CA_FIX_REACH, CA_FIX_REACH), d[:, 1:].round().clamp(-CA_FIX_REACH, CA_FIX_REACH)
    hp, wp = h + 2 * pad, w + 2 * pad
    planes = [m - a * gs, a, colour] + ([m] if clamp else [])
    src = torch.cat([F.pad(t, (pad,) * 4, value=-1.0 if t is colour else 0.0) for t in planes], 1)
    src = src.reshape(b, len(planes), hp * wp)
    y = torch.arange(h, device=m.device).view(1, 1, h, 1)
    x = torch.arange(w, device=m.device).view(1, 1, 1, w)
    base_idx = ((y + pad + ry) * wp + x + pad + rx).long().reshape(b, 1, h * w).expand(b, len(planes), h * w)
    gauss = lambda t: torch.exp(-t ** 2 / (2 * CA_FIX_SIGMA ** 2))
    wy = [gauss(ry + i - d[:, :1]) for i in range(-r, r + 1)]
    wx = [gauss(rx + j - d[:, 1:]) for j in range(-r, r + 1)]
    num, num_a, den = torch.zeros_like(m), torch.zeros_like(m), torch.zeros_like(m)
    lo, hi = torch.full_like(m, float('inf')), torch.full_like(m, -float('inf'))
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            v = src.gather(2, base_idx + (i * wp + j)).view(b, len(planes), h, w)
            wgt = wy[i + r] * wx[j + r] * (v[:, 2:3] == colour)
            num, num_a, den = num + wgt * v[:, :1], num_a + wgt * v[:, 1:2], den + wgt
            if clamp:
                near = wgt > CA_FIX_NEAR
                lo = torch.where(near, torch.minimum(lo, v[:, 3:4]), lo)
                hi = torch.where(near, torch.maximum(hi, v[:, 3:4]), hi)
    fixed = (num + num_a * g) / den.clamp(min=1e-12)
    if clamp:                                      # (where no sample is near, unbounded)
        fixed = torch.where(lo.isfinite(), torch.minimum(torch.maximum(fixed, lo), hi), fixed)
    return torch.where(colour != 1, fixed, m)


def midtone_shift(lin, to):
    """Move mid-grey (MIDTONE_PIVOT) to `to` times itself (B,1,1,1) by a power
    curve on luma, L -> L^g with g = log(pivot * to) / log(pivot), all three
    channels scaled alike (colour kept). Black and L = 1 stay where they are
    (values above 1, from the contrast, are shifted too and clip in `sensor`),
    so unlike an exposure it neither adds clipping nor lets in light; shadows
    below the pivot move more, in stops, than the highlights do."""
    luma = (lin * LUMA_SRGB.to(lin.device).view(1, 3, 1, 1)).sum(1, keepdim=True).clamp(min=1e-4)
    g = (MIDTONE_PIVOT * to).log() / math.log(MIDTONE_PIVOT)
    return lin * luma ** (g - 1)


def contrast_some(lin):
    """Raise the contrast of CONTRAST_P of the patches about their own level:
    log luma minus its mean over the patch, times a gain uniform in
    CONTRAST_GAIN, by scaling all three channels alike (colour kept, nothing
    negative). So a dark patch stays dark and a bright one bright on average,
    each a stop off its mean now `gain` stops off; what goes past SAT clips in
    `sensor`. Returns the patches and each one's gain (1 = unchanged). Drawn
    after everything else, so the other draws keep their random stream. All
    patches go through the same arithmetic, where gain 1 makes the factor
    exactly 1: picking the chosen ones (`nonzero`) made the host wait for the
    GPU and split the compiled batch, ~2.4 ms of an 8.5 ms batch."""
    n = lin.shape[0]
    dev = lin.device
    on = torch.rand(n, device=dev) < CONTRAST_P
    g = torch.rand(n, device=dev) * (CONTRAST_GAIN[1] - CONTRAST_GAIN[0]) + CONTRAST_GAIN[0]
    gain = torch.where(on, g, torch.ones_like(g)).view(n, 1, 1, 1)
    luma = (lin * LUMA_SRGB.to(dev).view(1, 3, 1, 1)).sum(1, keepdim=True).clamp(min=1e-4)
    ll = luma.log()
    factor = ((gain - 1) * (ll - ll.mean((2, 3), keepdim=True))).exp()
    factor = factor.clamp(1 / CONTRAST_LIMIT, CONTRAST_LIMIT)    # a bright pixel in a dark patch got 1e8x
    return lin * factor, gain


def blur_every(x):
    """Gaussian-blur every BLUR_EVERY-th patch, sigma uniform in (0, BLUR_MAX_SIGMA]
    px. The other patches are left alone (they were once blurred with an
    exact delta kernel, the same result at 8x the cost)."""
    n, c, h, w = x.shape
    s = torch.rand(n, device=x.device) * BLUR_MAX_SIGMA
    idx = torch.arange(0, n, BLUR_EVERY, device=x.device)
    s, m = s[idx].clamp(min=1e-3), len(idx)
    r = math.ceil(3 * BLUR_MAX_SIGMA)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k[None] ** 2 / (2 * s[:, None] ** 2))
    g = (g / g.sum(1, keepdim=True)).repeat_interleave(c, 0)        # m*c, K
    y = F.pad(x[idx].reshape(1, m * c, h, w), (r, r, r, r), mode='reflect')
    y = F.conv2d(y, g[:, None, None, :], groups=m * c)
    y = F.conv2d(y, g[:, None, :, None], groups=m * c)
    out = x.clone()
    out[idx] = y.reshape(m, c, h, w)
    return out


# ------------------------------------------------------------------- the loss

def tone(x):
    """Signed display gamma 1/2.2 with a linear toe below sRGB's knee, value and
    slope continuous there; odd-symmetric so out-of-gamut values stay finite."""
    g, k = 1 / 2.2, 0.0031308
    c = k ** g * (1 - g) / (1 - k ** g * (1 - g))
    s = (1 + c) * g * k ** (g - 1)
    ax = x.abs()
    y = torch.where(ax < k, ax * s, (1 + c) * ax.clamp(min=k).pow(g) - c)
    return y * x.sign()


def tone_error(out, target, srgb_from_cam, border):
    """tone(out) - tone(target) in linear sRGB, `border` px dropped."""
    b = border
    return (tone(apply_ccm(out.float(), srgb_from_cam)) -
            tone(apply_ccm(target.float(), srgb_from_cam)))[..., b:-b, b:-b]


def charbonnier(out, target, srgb_from_cam, border):
    """Charbonnier in tone-mapped linear sRGB, `border` px dropped."""
    d = tone_error(out, target, srgb_from_cam, border)
    return (d * d + LOSS_EPS).sqrt().mean()


def pattern_loss(d, period):
    """The RMS over positions of the error's mean per position of the CFA
    period (y mod period, x mod period), less its mean over positions, per
    patch and channel, of a tone_error d (whose border must be a multiple of
    the period); the mean over patches."""
    ph = d.unflatten(-1, (-1, period)).unflatten(-3, (-1, period)).mean((-4, -2))   # B, 3, P, P
    ph = ph - ph.mean((-2, -1), keepdim=True)
    return (ph.pow(2).mean((1, 2, 3)) + 1e-12).sqrt().mean()


def chroma_loss(d):
    """Charbonnier of the colour error of a tone_error d: its R - G and B - G.
    The error common to the channels (luma, and the noise that goes with it)
    cancels, what is left is false colour and the CFA's chroma pattern."""
    c = torch.stack([d[:, 0] - d[:, 1], d[:, 2] - d[:, 1]], 1)
    return (c * c + LOSS_EPS).sqrt().mean()


def gradient_loss(d):
    """Charbonnier of the horizontal and vertical differences of a tone_error
    d, which are the differences of the image's gradients, out minus target:
    the error in the fine detail and the edges."""
    gx = d[..., :, 1:] - d[..., :, :-1]
    gy = d[..., 1:, :] - d[..., :-1, :]
    return ((gx * gx + LOSS_EPS).sqrt().mean() + (gy * gy + LOSS_EPS).sqrt().mean()) / 2


def loss(out, target, srgb_from_cam, border, period, chroma=0.0, gradient=0.0):
    """The training loss: charbonnier + PATTERN_WEIGHT x pattern_loss, on one
    tone_error, plus `chroma` x chroma_loss and `gradient` x gradient_loss
    when they are not 0 (finetune.py's; with 0 the loss is as it was, bit for
    bit). train() compiles it: computing the error once and fusing it
    took the loss from 2.8 to 0.5 ms and its backward 2.4 ms shorter (X-Trans,
    batch 32: 1319 -> 1640 img/s)."""
    d = tone_error(out, target, srgb_from_cam, border)
    total = (d * d + LOSS_EPS).sqrt().mean() + PATTERN_WEIGHT * pattern_loss(d, period)
    if chroma:
        total = total + chroma * chroma_loss(d)
    if gradient:
        total = total + gradient * gradient_loss(d)
    return total


def to_lab(rgb):
    """B,3,H,W linear sRGB -> CIE Lab (D65), L in 0..100."""
    xyz = apply_ccm(rgb, SRGB_TO_XYZ.to(rgb).expand(rgb.shape[0], 3, 3))
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device).view(1, 3, 1, 1)
    f = torch.where(xyz > (6 / 29) ** 3, xyz.clamp(min=1e-6).pow(1 / 3), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return torch.cat([116 * f[:, 1:2] - 16, 500 * (f[:, :1] - f[:, 1:2]), 200 * (f[:, 1:2] - f[:, 2:])], 1)


def moire_gain(out, target, srgb_from_cam, border):
    """B: the largest value of the moire gain map of each patch (`border` px
    dropped). The map is, per Lab channel and 2D frequency w, log((|F_out(w)|^2
    + eta) / (|F_target(w)|^2 + eta)) for |w| <= MOIRE_R * pi and 0 above,
    blurred (MOIRE_SIGMA bins, wrapping), as in Gharbi et al. 2016 (eq. 7):
    energy the demosaic added where the image had none, as false colour or
    aliasing does."""
    b = border
    lab = lambda z: to_lab(apply_ccm(z.float(), srgb_from_cam))[..., b:-b, b:-b]
    fi = torch.fft.fft2(lab(target), norm='ortho')
    fo = torch.fft.fft2(lab(out), norm='ortho')
    gain = (((fo.real ** 2 + fo.imag ** 2 + MOIRE_ETA) / (fi.real ** 2 + fi.imag ** 2 + MOIRE_ETA))).log()
    h, w = gain.shape[-2:]
    wy = torch.fft.fftfreq(h, device=gain.device).view(h, 1) * 2 * math.pi
    wx = torch.fft.fftfreq(w, device=gain.device).view(1, w) * 2 * math.pi
    gain = gain * ((wy ** 2 + wx ** 2).sqrt() <= MOIRE_R * math.pi)
    r = math.ceil(3 * MOIRE_SIGMA)
    k = torch.arange(-r, r + 1, device=gain.device, dtype=gain.dtype)
    k = torch.exp(-k ** 2 / (2 * MOIRE_SIGMA ** 2))
    k = k / k.sum()
    gain = F.pad(gain, (r, r, r, r), mode='circular')               # frequency is periodic
    gain = F.conv2d(gain.reshape(-1, 1, *gain.shape[-2:]), k.view(1, 1, 1, -1))
    gain = F.conv2d(gain, k.view(1, 1, -1, 1)).reshape(-1, 3, h, w)
    return gain.amax((1, 2, 3))


@torch.no_grad()
def mine_scores(net, crops, cfa, border):
    """(loss, moire gain), each B, of uint8 crops (B,3,S,S) through a fixed
    sensor with no noise (camera matrix 0, MINE_WB, no exposure change, no contrast) and the
    network (train passes its compiled one, ~10 ms per batch of 32 against 22 eager
    with the head): the content's own difficulty, whatever the training draws
    would add. The loss is the training one's Charbonnier per patch."""
    n, dev = crops.shape[0], crops.device
    lin = srgb_to_linear(crops.float() / 255)
    cam = CAM_FROM_SRGB_BANK[:1].to(dev).expand(n, 3, 3)
    gain = torch.tensor([MINE_WB[0], 1.0, MINE_WB[1]], device=dev).view(1, 3, 1, 1)
    one, zero = torch.ones(n, 1, 1, 1, device=dev), torch.zeros(n, 1, 1, 1, device=dev)
    noisy, sigma, target, srgb_from_cam = sensor(
        lin, cfa, cam, torch.linalg.inv(cam), gain, one, zero, zero, zero, zero)
    with torch.autocast('cuda', dtype=torch.float16):
        out = net(noisy.contiguous(memory_format=torch.channels_last),
                  sigma.contiguous(memory_format=torch.channels_last))
    d = tone_error(out, target, srgb_from_cam, border)
    return (d * d + LOSS_EPS).sqrt().mean((1, 2, 3)), moire_gain(out, target, srgb_from_cam, border)


def psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean())


# ------------------------------------------------------------------- the data

class Crops(torch.utils.data.Dataset):
    """Random size x size crops of the training images, uniform over their
    pixels, under a random rotation / flip. Returns 3,size,size uint8; with
    `with_origin` also the crop's (image, y, x) as a 3-vector of int64, for
    `cut` to cut it again (hard-patch mining)."""

    def __init__(self, data, size, length, with_origin=False):
        self.data, self.size, self.length, self.with_origin = data, size, length, with_origin
        self.index = np.load(str(data)[:-4] + '_index.npy')         # offset, h, w per image
        self.index = self.index[:-HOLDOUT_IMAGES]
        h, w = self.index[:, 1], self.index[:, 2]
        self.cum = np.cumsum((h - size + 1) * (w - size + 1)).tolist()  # crop origins
        self.flat = None                                            # opened per worker

    def __len__(self):
        return self.length

    def cut(self, k, y, x):
        """The crop of image k whose top left is (y, x), as it is, 3,size,size uint8."""
        if self.flat is None:
            self.flat = np.load(self.data, mmap_mode='r')
        o, h, w = (int(v) for v in self.index[k])
        im = self.flat[o:o + h * w * 3].reshape(h, w, 3)
        return torch.from_numpy(np.array(im[y:y + self.size, x:x + self.size])).permute(2, 0, 1)

    def __getitem__(self, i):
        k = random.choices(range(len(self.index)), cum_weights=self.cum)[0]
        _, h, w = (int(v) for v in self.index[k])
        y, x = random.randrange(h - self.size + 1), random.randrange(w - self.size + 1)
        crop = torch.rot90(self.cut(k, y, x), random.randrange(4), [-2, -1])
        if random.random() < 0.5:
            crop = crop.flip(-1)
        crop = crop.contiguous()
        return (crop, torch.tensor([k, y, x])) if self.with_origin else crop


def held_out_image(data, k):
    """Held-out image k (negative = from the end) as h,w,3 uint8."""
    index = np.load(str(data)[:-4] + '_index.npy')
    o, h, w = (int(v) for v in index[k])
    return np.array(np.load(data, mmap_mode='r')[o:o + h * w * 3].reshape(h, w, 3))


def eval_set(data, size, cfa, dev, noise=None):
    """EVAL_PATCHES fixed crops of the held-out images through random_batch
    (`noise` as there), under a fixed seed: the same tensors at every
    checkpoint."""
    index = np.load(str(data)[:-4] + '_index.npy')
    flat = np.load(data, mmap_mode='r')
    rng = np.random.default_rng(EVAL_SEED)
    crops = []
    for k in rng.integers(len(index) - HOLDOUT_IMAGES, len(index), EVAL_PATCHES):
        o, h, w = (int(v) for v in index[k])
        y, x = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
        im = flat[o:o + h * w * 3].reshape(h, w, 3)
        crops.append(torch.from_numpy(np.array(im[y:y + size, x:x + size])).permute(2, 0, 1))
    crops = torch.stack(crops).to(dev)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(EVAL_SEED)
        return [random_batch(c, cfa, noise=noise) for c in crops.split(32)]


@torch.no_grad()
def evaluate(net, batches, border):
    """(loss, psnr) on the eval set, `border` px dropped per side from both;
    psnr in linear sRGB clamped to [0, 1]. (The psnr counted the border until
    2026-10-18, when the padding became zeros: its numbers are higher now.)"""
    loss, p = 0.0, 0.0
    b = border
    for noisy, sigma, target, srgb_from_cam in batches:
        with torch.autocast('cuda', dtype=torch.float16):
            out = net(noisy, sigma)
        loss += charbonnier(out, target, srgb_from_cam, border).item()
        srgb = lambda z: apply_ccm(z.float(), srgb_from_cam).clamp(0, 1)[..., b:-b, b:-b]
        p += psnr(srgb(out), srgb(target)).item()
    return loss / len(batches), p / len(batches)


# --------------------------------------------------------------- the training

def train(build, cfa, tile, ckpt, steps, batch, patch, lr, ema_decay, clip_grad,
          data, border, halo, noise=None, mine=0.0, chroma=0.0, gradient=0.0):
    """Train build() for `steps` steps. `ckpt` holds everything, and a run picks
    up from it if it exists:

        model   the EMA weights: what inference and the export use
        raw     the weights being trained
        opt     AdamW state
        scaler  fp16 GradScaler state
        step    steps done

    Saved every 10000 steps and at the end, each save printing the held-out
    PSNR; at the end the EMA weights are exported for the wgpu runner.
    `border`: px per side the loss drops; `halo`: see export; `noise`: see
    random_batch (the held-out set and the export's test vector follow it);
    `mine`: the share of a batch hard-patch mining replaces at the last step
    (0 = off, see MINE_START); `chroma`, `gradient`: the weights of chroma_loss and gradient_loss (see loss).
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'
    cfa = cfa.to(dev)

    net = build().to(dev).to(memory_format=torch.channels_last)
    ema = build().to(dev).to(memory_format=torch.channels_last).eval().requires_grad_(False)
    ema.load_state_dict(net.state_dict())
    # The convs pad with zeros (conv()): the loss and the PSNR drop `border` px
    # per side, which must be at least the reach (halo is checked against it
    # by the export), or the padding would show in them
    assert border >= halo, f'border {border} px < halo {halo} px: zero padding would show'
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=True)
    scaler = torch.amp.GradScaler('cuda')
    start = 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=dev)
        net.load_state_dict(ck['raw'])
        ema.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        start = ck['step']
        print(f'resuming {ckpt} at step {start}')
    print(f'{sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters')

    def save(step):
        torch.save({'model': ema.state_dict(), 'raw': net.state_dict(),
                    'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': step},
                   str(ckpt) + '.tmp')
        os.replace(str(ckpt) + '.tmp', ckpt)       # never leave a half-written file
        loss, p = evaluate(ema, held_out, border)
        print(f'{step:7d}  saved {ckpt.name}   held-out: loss {loss:.5f}  psnr {p:.2f} dB')

    held_out = eval_set(data, patch, cfa, dev, noise)
    loader = torch.utils.data.DataLoader(
        Crops(data, patch, (steps - start) * batch, mine > 0),
        batch_size=batch,
        num_workers=WORKERS, pin_memory=True, drop_last=True, persistent_workers=True)
    fnet = torch.compile(net)
    fbatch = torch.compile(random_batch)
    floss = torch.compile(loss)
    ema_p, net_p = list(ema.parameters()), list(net.parameters())
    if mine:                                  # the buffer of hard crops, a ring
        csize = patch
        hard = torch.zeros(MINE_BUFFER, 3, dtype=torch.int64)       # (image, y, x) of each
        reader = Crops(data, csize, 0)                              # cuts them again
        n_hard = ring = seen = flagged = 0
        mean_loss = None

    t0, run_loss = time.time(), torch.zeros((), device=dev)
    for i, crops in enumerate(loader, start):
        if mine:
            crops, origin = crops
        # linear warm-up, then cosine decay to 0 at the last step
        for g in opt.param_groups:
            g['lr'] = lr * min(1.0, i / WARMUP) * 0.5 * (1 + math.cos(math.pi * i / steps))
        crops = crops.to(dev, non_blocking=True)
        if mine and i >= MINE_SCORE_START * steps and i % MINE_EVERY == 0:
            loss_p, moire = mine_scores(fnet, crops, cfa, border)
            mean_loss = loss_p.mean() if mean_loss is None else 0.9 * mean_loss + 0.1 * loss_p.mean()
            keep = ((loss_p > MINE_LOSS_RATIO * mean_loss) | (moire > MOIRE_T)).cpu()
            seen, flagged = seen + len(crops), flagged + int(keep.sum())
            at = (ring + torch.arange(int(keep.sum()))) % MINE_BUFFER
            hard[at] = origin[keep]
            ring, n_hard = (ring + len(at)) % MINE_BUFFER, min(MINE_BUFFER, n_hard + len(at))
        if mine and n_hard and i >= MINE_START * steps:
            share = mine * (i / steps - MINE_START) / (1 - MINE_START)
            span, again = MINE_JITTER, []
            for k in torch.randint(n_hard, (round(share * len(crops)),)).tolist():
                img, y, x = hard[k].tolist()
                h, w = int(reader.index[img, 1]), int(reader.index[img, 2])
                y = min(max(y + random.randint(-span, span), 0), h - csize)
                x = min(max(x + random.randint(-span, span), 0), w - csize)
                c = torch.rot90(reader.cut(img, y, x), random.randrange(4), [-2, -1])
                again.append(c.flip(-1) if random.random() < 0.5 else c)
            if again:
                crops[:len(again)] = torch.stack(again).to(dev)
        noisy, sigma, target, srgb_from_cam = fbatch(crops, cfa, noise=noise)
        noisy = noisy.contiguous(memory_format=torch.channels_last)
        sigma = sigma.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.float16):
            out = fnet(noisy, sigma)
        step_loss = floss(out, target, srgb_from_cam, border, cfa.shape[0], chroma, gradient)

        opt.zero_grad(set_to_none=True)
        scaler.scale(step_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net_p, clip_grad)
        scaler.step(opt)
        scaler.update()
        # EMA; the decay ramps up over the first steps so the average does not
        # remember the random init
        d = min(ema_decay, (1 + i) / (10 + i))
        with torch.no_grad():
            torch._foreach_lerp_(ema_p, net_p, 1 - d)

        run_loss += step_loss.detach()
        if (i + 1) % 1000 == 0:
            print(f'{i + 1:7d}  loss {run_loss.item() / 1000:.5f}  '
                  f'lr {opt.param_groups[0]["lr"]:.2e}  '
                  f'{(i + 1 - start) * batch / (time.time() - t0):.0f} img/s'
                  + (f'  hard {n_hard} (flagged {flagged / seen:.0%})' if mine and seen else ''))
            run_loss.zero_()
            if mine:
                seen = flagged = 0
        if (i + 1) % 10000 == 0:
            save(i + 1)
    if steps % 10000 and start < steps:
        save(steps)
    export(build, cfa, tile, ckpt, data, halo, noise)


# -------------------------------------------------------------- wgpu export

@torch.no_grad()
def export(build, cfa, tile, ckpt, data, halo, noise=None):
    """ckpt's EMA weights -> <ckpt>_wgpu/ for the Rust executor in wgpu-run/:

    f16/, f32/   graph.txt + one .wgsl per dispatch + weights.bin, the network
                 compiled for WGPU_WINDOW x WGPU_WINDOW px windows (see wgsl.py);
                 f16 needs the adapter's SHADER_F16 feature, f32 runs anywhere
    test.bin     u32 h, u32 w, f32[9] srgb_from_cam, then f32 mosaic[h*w],
                 sigma[h*w], fp32 PyTorch output[3*h*w] (balanced camera space),
                 clean[3*h*w] (linear sRGB): a held-out image through a fixed
                 sensor

    halo: context px per window side, at least the network's reach and a
    multiple of the CFA period. noise: as in random_batch; the test vector's
    a and b are then the geometric middles of its ranges, not TEST_A / TEST_B.
    """
    dev = 'cuda'
    cfa = cfa.to(dev)
    net = build().to(dev).eval()
    net.load_state_dict(torch.load(ckpt, map_location=dev)['model'])
    out = Path(ckpt).with_name(Path(ckpt).stem + '_wgpu')

    # weights rounded to fp16 once, so the f16 and f32 builds compute with the same values
    arr = lambda t: t.detach().half().float().cpu().numpy()
    # The trunk as convs: (name, weight, bias, LeakyReLU after, LeakyReLU on
    # the input, input added to the output). A LeakyReLU right after a conv is
    # its output's; one after a Residual (whose output must stay as it is) is
    # the next conv's input's.
    mods = list(net.net.children())
    trunk, act_in = [], False
    for j, m in enumerate(mods):
        if isinstance(m, nn.Conv2d):
            trunk.append([f'net.{j}', arr(m.weight), arr(m.bias), False, act_in, False])
            act_in = False
        elif isinstance(m, Residual):
            trunk.append([f'net.{j}.conv', arr(m.conv.weight), arr(m.conv.bias), False, True, True])
        elif isinstance(m, nn.LeakyReLU):
            if j and isinstance(mods[j - 1], nn.Conv2d):
                trunk[-1][3] = True
            else:
                act_in = True
    bank = lambda m: (arr(m.conv.weight), arr(m.conv.bias), m.per ** 2) \
        if isinstance(m, PhaseConv) else (arr(m.weight), arr(m.bias), 1)
    h = net.head
    layers = {'trunk': [tuple(t) for t in trunk], 'pack': net.layout,
              'head_in': h and bank(h[0]), 'head_out': h and bank(h[2]),
              'head_residual': net.head_residual}
    # Receptive-field radius: each trunk 3x3 reaches one cell, and a pixel can
    # sit at the far side of its own: tile * (k3 + 1) - 1; a head's 3x3 one px
    # more.
    k3 = sum(t[1].shape[-1] == 3 for t in trunk)
    period = cfa.shape[0]
    reach = tile * (k3 + 1) - 1 + (h is not None)
    assert halo >= reach and halo % period == 0, \
        f'halo {halo}: the reach is {reach} px, the period {period}'
    for precision in ('f16', 'f32'):
        n_ops, nbytes = wgsl.write(out / precision, precision, layers, cfa.cpu().numpy(),
                                   tile, WGPU_WINDOW, halo)
    print(f'exported {out}/: {n_ops} dispatches per {WGPU_WINDOW} px window, halo {halo} px, '
          f'{nbytes / 1e6:.0f} MB of f32 buffers')

    h, w = TEST_SIZE
    im = torch.from_numpy(held_out_image(data, TEST_IMAGE)[:h, :w]).permute(2, 0, 1)[None]
    lin = srgb_to_linear(im.to(dev).float() / 255)
    torch.manual_seed(0)
    test_a, test_b = (TEST_A, TEST_B) if noise is None else (math.sqrt(lo * hi) for lo, hi in noise)
    noisy, sigma, srgb_from_cam, ref = fixed_sensor(net, lin, cfa, TEST_WB, 1.0, test_a, test_b)
    with open(out / 'test.bin', 'wb') as f:
        f.write(struct.pack('<II', h, w))
        for t in (srgb_from_cam, noisy, sigma, ref, lin):
            f.write(t.float().cpu().numpy().astype('<f4').tobytes())


def radial_ca(h, w, red, blue, dev):
    """1,4,h,w: the lateral CA field of an h x w frame, red's (dy, dx) then
    blue's against green at each pixel, px: s (p - c), c the centre (the
    channel at p + s (p - c) is green's at p, src/lens/measure.rs's
    convention), red and blue shifted by `red` and `blue` px at the corner,
    s constant."""
    half = math.hypot(h, w) / 2
    y = (torch.arange(h, device=dev).view(1, 1, h, 1) - (h - 1) / 2).expand(1, 1, h, w)
    x = (torch.arange(w, device=dev).view(1, 1, 1, w) - (w - 1) / 2).expand(1, 1, h, w)
    return torch.cat([red / half * y, red / half * x, blue / half * y, blue / half * x], 1)


def fixed_sensor(net, lin, cfa, wb, e, a, b, autocast=False):
    """One linear-sRGB image through the sensor with fixed settings (camera
    matrix 0 of the bank, the given WB, exposure and noise, sigma told the true
    a and b), then through the network. -> (noisy, sigma, srgb_from_cam, out)."""
    dev = lin.device
    one = lambda v: torch.full((1, 1, 1, 1), float(v), device=dev)
    cam_from_srgb = CAM_FROM_SRGB_BANK[:1].to(dev)
    gain = torch.tensor([wb[0], 1.0, wb[1]], device=dev).view(1, 3, 1, 1)
    noisy, sigma, _, srgb_from_cam = sensor(
        lin, cfa, cam_from_srgb, torch.linalg.inv(cam_from_srgb), gain,
        one(e), one(a), one(b), one(a), one(b))
    with torch.autocast('cuda', dtype=torch.float16, enabled=autocast):
        out = net(noisy, sigma).float()
    return noisy, sigma, srgb_from_cam, out


# ------------------------------------------------------------------ inference

INFER_BORDER = 8   # px dropped per side before PSNR
INFER_PAD = 48     # px the photo is mirrored by on every side for the network and cropped off again:
                   # zero padding, which a pixel sees from no further than the reach (36 at most),
                   # would show at the edges. A multiple of every period, so the CFA phase holds


@torch.no_grad()
def infer(build, cfa, tile, ckpt, image, a, b, exposure, wb, out_path):
    """Mosaic + noise a clean photo, demosaic it, print PSNR, write out_path."""
    dev = 'cuda'
    cfa = cfa.to(dev)
    net = build().to(dev).eval()
    net.load_state_dict(torch.load(ckpt, map_location=dev)['model'])
    im = Image.open(image).convert('RGB')
    k = math.lcm(tile, cfa.shape[0])
    w, h = im.width // k * k, im.height // k * k
    im = torch.from_numpy(np.array(im.crop((0, 0, w, h)))).permute(2, 0, 1)[None]
    lin = srgb_to_linear(im.to(dev).float() / 255)

    q = INFER_PAD
    mirrored = F.pad(lin, (q,) * 4, mode='reflect')
    _, _, srgb_from_cam, out = fixed_sensor(net, mirrored, cfa, wb, exposure, a, b, autocast=True)
    out = apply_ccm(out, srgb_from_cam).clamp(0, 1)[..., q:-q, q:-q]
    r = INFER_BORDER
    p = psnr(out[..., r:-r, r:-r], lin.clamp(0, 1)[..., r:-r, r:-r]).item()
    print(f'{w}x{h}  a={a:g} b={b:g} exposure={exposure:g}  '
          f'psnr {p:.2f} dB')
    arr = (linear_to_srgb(out[0]).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    Image.fromarray(arr).save(out_path)
    print('wrote', out_path)
