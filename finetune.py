"""Fine-tune a trained model with the current training (loss, data), for
FINETUNE_STEPS at FINETUNE_LR, into <name><FINETUNE_SUFFIX>.pt (and its export
<name><FINETUNE_SUFFIX>_wgpu/), leaving the model itself alone. To promote the
result, save it over the original with the original's 'step' (else train_*.py
would train it on) and export that.

The fine-tune adds a chroma error (R - G, B - G) and a gradient error to the
loss, at FINETUNE_CHROMA and FINETUNE_GRADIENT times their size (common.loss).
Both 0 gives the loss training has. For a control that separates what the
fine-tune does by itself from what the two terms do, run it once with both 0
and another FINETUNE_SUFFIX; suite.py shows every suffix it finds.

    python finetune.py train_xtrans          # xtrans.pt -> xtrans_ftl.pt
    python finetune.py train_xtrans train_bayer
    python finetune.py train_xtrans --data /tmp/div2k.npy   # prep.py output elsewhere

A fresh optimizer and a new warm-up and cosine schedule from step 0; an
existing <name>_ft.pt is resumed instead.
"""
import argparse
import importlib

import torch

from common import DATA, WEIGHT_DECAY, train

FINETUNE_STEPS = 30_000
FINETUNE_LR = 4e-5
FINETUNE_SUFFIX = '_ftl'      # the output is <name>_ftl.pt: the older _ft.pt files are left alone
FINETUNE_CHROMA = 0.25        # weight of the R - G, B - G error (about 30% of the loss at 0.016 vs 0.013)
FINETUNE_GRADIENT = 0.25      # weight of the gradient error (about 12% of the loss)


def finetune(script, data=DATA):
    mod = importlib.import_module(script)
    ft = mod.CKPT.with_name(mod.CKPT.stem + FINETUNE_SUFFIX + '.pt')
    if not ft.exists():
        net = mod.build()
        ck = torch.load(mod.CKPT, map_location='cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY, fused=True)
        torch.save({'model': ck['model'], 'raw': ck['raw'], 'opt': opt.state_dict(),
                    'scaler': torch.amp.GradScaler('cuda').state_dict(), 'step': 0}, ft)
        print(f'{ft.name}: from {mod.CKPT.name} (step {ck["step"]})')
    train(mod.build, mod.CFA, mod.TILE, ft, FINETUNE_STEPS, mod.BATCH, mod.PATCH, FINETUNE_LR,
          mod.EMA, mod.CLIP_GRAD, data, mod.BORDER, mod.HALO, getattr(mod, 'NOISE', None),
          getattr(mod, 'MINE', 0.0), FINETUNE_CHROMA, FINETUNE_GRADIENT)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('scripts', nargs='+', help='train_*.py module names')
    ap.add_argument('--data', default=DATA, help='prep.py output')
    args = ap.parse_args()
    for name in args.scripts:
        finetune(name, args.data)
