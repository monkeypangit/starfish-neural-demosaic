"""Writes the CA benchmark's test vectors into ca_bench/: synthetic scenes through
the sensor with a known lateral CA, plus the same mosaic without CA (the
truth), in ca_fix.bin's layout (src/processing/ca_fix.rs, `ca_benchmark`).

    python ca_bench.py [xtrans|bayer]

Scenes (linear sRGB, values above 1 clip at the sensor; the sensor's red and
blue clip at balanced 2.0 / 1.6 with TEST_WB, green at 1, so lamps between
1 and 2 clip in some channels only):
  lamps         12 discs (neutral, warm, cool, red; 1.5-20x white) with a glow,
                over a dark sky, blurred by the lens (sigma LENS_BLUR)
  edges         saturated colour bars and a ramp, below and above the clip
  natural       kodim05, detail without clipping
  natural_lamps kodim05 with three lamps over it
Each with no noise and with noise (NOISE_A, NOISE_B).
"""
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import (BAYER, CAM_FROM_SRGB_BANK, HERE, TEST_WB, XTRANS, ca_correct, cfa_masks, radial_ca, sensor,
                    srgb_to_linear)

CA = (3.0, -3.5)         # red's and blue's shift at the corner, px (TEST_CA is 2, -2.5)
LENS_BLUR = 0.7          # the lens' blur, px: sets how sharp a clipped edge is
NOISE_A, NOISE_B = 5e-4, 1e-6
SIZE = 768
OUT = HERE / 'ca_bench'


def blur(x, sigma):
    r = int(3 * sigma) + 1
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device).float() ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def lamp(lin, cy, cx, r, rgb, level, glow_amount=0.04):
    """A disc of linear colour rgb * level (a hair of edge softness) and a glow."""
    h, w = lin.shape[-2:]
    y = torch.arange(h, device=lin.device).view(h, 1).float()
    x = torch.arange(w, device=lin.device).view(1, w).float()
    d = ((y - cy) ** 2 + (x - cx) ** 2).sqrt()
    disc = (r + 0.5 - d).clamp(0, 1)
    colour = torch.tensor(rgb, device=lin.device).view(3, 1, 1) * level
    glow = torch.exp(-d ** 2 / (2 * (r * 0.9) ** 2)) * glow_amount
    return lin + colour * (disc + glow)[None, None]


def lamps_scene(dev, base=None, only=None):
    h = w = SIZE
    if base is None:
        yy = torch.linspace(0, 1, h, device=dev).view(h, 1).expand(h, w)
        sky = torch.stack([0.010 + 0.01 * yy, 0.012 + 0.012 * yy, 0.025 + 0.02 * yy])
        lin = sky[None].clone()
    else:
        lin = base.clone()
    colours = [(1.0, 1.0, 1.0), (1.0, 0.70, 0.35), (0.70, 0.85, 1.0), (1.0, 0.25, 0.15)]
    levels = [1.5, 4.0, 20.0]
    cells = [(r, c) for r in range(4) for c in range(3)]
    for i, (row, col) in enumerate(cells):
        if only is not None and i not in only:
            continue
        cy = 90 + row * 195 + (30 if col == 1 else 0)
        cx = 110 + col * 270
        lin = lamp(lin, cy, cx, 20, colours[row], levels[col])
    return lin


def edges_scene(dev):
    h = w = SIZE
    lin = torch.full((1, 3, h, w), 0.02, device=dev)
    colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1), (1, 0.5, 0)]
    # vertical bars, left half dim (0.6), right half bright (3)
    bar = w // len(colours)
    for i, c in enumerate(colours):
        for level, (y0, y1) in ((0.6, (0, h // 2)), (3.0, (h // 2, h))):
            lin[0, :, y0 + 8:y1 - 8, i * bar + 8:(i + 1) * bar - 8] = torch.tensor(c, device=dev, dtype=torch.float32).view(3, 1, 1) * level
    return lin


def natural_scene(dev, name='kodim05'):
    im = np.array(Image.open(HERE.parent.parent / 'demosaic' / 'images' / (name + '.png')).convert('RGB'))
    h, w = SIZE, 510
    im = im[:h // 1 if False else 512][:h, :SIZE][:h]
    lin = srgb_to_linear(torch.from_numpy(im[:510, :768]).to(dev).permute(2, 0, 1)[None].float() / 255)
    return lin


def write(name, lin, cfa, noise, ca=CA, lens_blur=LENS_BLUR):
    dev = lin.device
    h, w = lin.shape[-2:]
    h, w = h // 6 * 6, w // 6 * 6
    lin = lin[..., :h, :w]
    lin = blur(lin, lens_blur)
    one = lambda v: torch.full((1, 1, 1, 1), float(v), device=dev)
    cam = CAM_FROM_SRGB_BANK[:1].to(dev)
    gain = torch.tensor([TEST_WB[0], 1.0, TEST_WB[1]], device=dev).view(1, 3, 1, 1)
    field = radial_ca(h, w, *ca, dev)
    a, b = (NOISE_A, NOISE_B) if noise else (0, 0)
    torch.manual_seed(1)
    m = sensor(lin, cfa, cam, torch.linalg.inv(cam), gain, one(1), one(a), one(b), one(a), one(b), field)[0]
    clean = sensor(lin, cfa, cam, torch.linalg.inv(cam), gain, one(1), one(0), one(0), one(0), one(0), None)[0]
    fixed = ca_correct(m, cfa, field, clamp=False)
    site_gain = (gain * cfa_masks(cfa, h, w).to(dev)).sum(1, keepdim=True)
    half = float(np.hypot(h, w) / 2)
    OUT.mkdir(exist_ok=True)
    with open(OUT / f'{name}{"_noisy" if noise else ""}.bin', 'wb') as f:
        f.write(struct.pack('<III', h, w, cfa.shape[0]))
        f.write(struct.pack('<ff', ca[0] / half, ca[1] / half))
        f.write(struct.pack('<fff', TEST_WB[0], 1.0, TEST_WB[1]))
        f.write(cfa.flatten().cpu().numpy().astype('<u4').tobytes())
        for t in (m / site_gain, fixed / site_gain, clean / site_gain):
            f.write(t.float().cpu().numpy().astype('<f4').tobytes())
    print(name, 'noisy' if noise else 'clean', h, w)


if __name__ == '__main__':
    dev = 'cuda'
    cfa = (XTRANS if (sys.argv[1:] or ['xtrans'])[0] == 'xtrans' else BAYER).to(dev)
    nat = natural_scene(dev)
    scenes = {
        'lamps': lamps_scene(dev),
        'edges': edges_scene(dev),
        'natural': nat,
        'natural_lamps': None,
    }
    # three lamps over the natural scene (a smaller, brighter set)
    over = nat.clone()
    h, w = nat.shape[-2:]
    for (cy, cx, level, rgb) in ((120, 150, 6.0, (1.0, 0.8, 0.5)), (300, 520, 2.0, (1, 1, 1)), (410, 260, 12.0, (0.8, 0.9, 1.0))):
        over = lamp(over, cy, cx, 18, rgb, level)
    scenes['natural_lamps'] = over
    for name, lin in scenes.items():
        for noise in (False, True):
            write(name, lin, cfa, noise)
    # stadium floodlights as in 1931359854.cr2: two rows of neutral lamps of
    # about 40 px, far above the clip, a 1% halo, the CA the file measured
    # (red 0.4, blue 1.6 px at the corner); sharper and softer lens
    flood = lamps_scene(dev, only=set())
    for row, cy in enumerate((180, 228)):
        for i in range(7):
            flood = lamp(flood, cy + (6 if i % 2 else 0), 90 + i * 46 + row * 20, 20, (1, 1, 1), 30.0, 0.013)
    for name, blur_ in (('floodlights', 0.7), ('floodlights_soft', 1.4)):
        for noise in (False, True):
            write(name, flood, cfa, noise, ca=(0.4, -1.6), lens_blur=blur_)
