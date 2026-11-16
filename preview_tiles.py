"""Write what the network is trained on as images: random training tiles, the
noisy raw mosaic shown as sRGB.

    python preview_tiles.py     # -> preview/*.png and preview/index.html

Tiles come from `Crops` through `random_batch`, as in training. The mosaic only
has one channel per site, so each pixel shows its own channel's noisy value and
the other two from the clean (clipped) target: the noise is exactly what the
network sees, at full resolution, with no demosaic in between. The image is
shown as simulated (clipping at SAT included).
"""
import html

import torch

from common import BAYER, DATA, HERE, XTRANS, Crops, cfa_masks, linear_to_srgb, random_batch, apply_ccm
from PIL import Image

CFAS = {'xtrans': XTRANS, 'bayer': BAYER}
COUNT = 1024                   # tiles per CFA
PATCH = 144                  # tile size, a multiple of 6
SCALE = 1                    # nearest-neighbour enlargement, so single pixels show
SEED = 0
OUT = HERE / 'preview'


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    crops = Crops(DATA, PATCH, COUNT)
    rows = []
    for name, cfa in CFAS.items():
        clean = torch.stack([crops[i] for i in range(COUNT)]).cuda()
        info = {}
        noisy, _, target, srgb_from_cam = random_batch(clean, cfa.cuda(), info)
        masks = cfa_masks(cfa, PATCH, PATCH).to(noisy)
        shown = target * (1 - masks) + noisy * masks               # B,3,H,W balanced camera
        rgb = linear_to_srgb(apply_ccm(shown, srgb_from_cam * info['e'].view(-1, 1, 1)).clamp(0, 1))
        for i in range(COUNT):
            f = f'{name}_{i:02d}.png'
            px = (rgb[i].permute(1, 2, 0) * 255 + 0.5).byte().cpu().numpy()
            Image.fromarray(px).resize((PATCH * SCALE,) * 2, Image.NEAREST).save(OUT / f)
            v = {k: t[i].flatten().tolist() for k, t in info.items()}
            blurred = i % 8 == 0                                    # blur_every
            noisy_p = v['a'][0] > 0
            cap = (f"{name} {i}: midtone {torch.tensor(v['midtone']).log2().item():+.1f} EV, "
                   f"a {v['a'][0]:.1e}, b {v['b'][0]:.1e}, noise-map error a x{v['sa'][0] / v['a'][0]:.2f} "
                   f"b x{v['sb'][0] / v['b'][0]:.2f}, " if noisy_p else
                   f"{name} {i}: midtone {torch.tensor(v['midtone']).log2().item():+.1f} EV, no noise, ")
            cap += f"WB R {v['gain'][0]:.2f} B {v['gain'][2]:.2f}"
            cap += f", saturation x{v['boost'][0]:.2f}" * (v['boost'][0] != 1)
            cap += f", contrast x{v['contrast'][0]:.2f}" * (v['contrast'][0] != 1)
            cap += ', R/B swapped' * bool(v['swap'][0]) + ', blurred' * blurred
            rows.append((f, cap))
    body = ''.join(f'<figure><img src="{f}"><figcaption>{html.escape(c)}</figcaption></figure>'
                   for f, c in rows)
    (OUT / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Training tiles</title>'
        '<style>body{background:#222;color:#ccc;font:12px sans-serif}'
        'figure{display:inline-block;margin:6px}img{image-rendering:pixelated;display:block}'
        f'figcaption{{max-width:{PATCH * SCALE}px}}</style>' + body)
    print(f'{len(rows)} tiles in {OUT}')


if __name__ == '__main__':
    main()
