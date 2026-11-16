"""Write the crops hard-patch mining would pick as images: random training crops
scored by a trained model (common.mine_scores) and the flagged ones saved.

    python preview_mined.py xtrans     # -> preview/mined/*.png and preview/mined/index.html
    python preview_mined.py bayer

A crop is flagged as in train(): its loss over MINE_LOSS_RATIO x the mean loss of the
scored crops, or its moire gain over MOIRE_T. Each image is the clean crop (the
scoring sensor has no noise), a caption gives the loss ratio and moire gain.
The mean is the whole sample's, where training keeps a running one, and the weights
are the finished checkpoint's, not the half-trained ones mining scores in a run, so
the flagged set is similar to what a run replays, not the same.
"""
import argparse
import html
import importlib

import numpy as np
import torch
from PIL import Image

from common import DATA, HERE, MINE_LOSS_RATIO, MOIRE_T, Crops, mine_scores

COUNT = 4096                   # crops scored
SHOW = 200                     # flagged crops written, the most severe first
SEED = 0
BATCH = 32
SCALE = 1                      # nearest-neighbour enlargement
OUT = HERE / 'preview' / 'mined'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['xtrans', 'bayer'], help='train_<model>.py and its checkpoint')
    ap.add_argument('--data', default=DATA, help='prep.py output')
    args = ap.parse_args()
    mod = importlib.import_module(f'train_{args.model}')
    dev = 'cuda'
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('*.png'):
        old.unlink()

    cfa = mod.CFA.to(dev)
    net = mod.build().to(dev).to(memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(mod.CKPT, map_location=dev)['model'])

    crops = Crops(args.data, mod.PATCH, COUNT)
    cut = torch.stack([crops[i] for i in range(COUNT)])
    loss, moire = [], []
    for i in range(0, COUNT, BATCH):
        l, m = mine_scores(net, cut[i:i + BATCH].to(dev), cfa, mod.BORDER)
        loss.append(l.cpu())
        moire.append(m.cpu())
    loss, moire = torch.cat(loss), torch.cat(moire)
    ratio = loss / loss.mean()
    by_loss, by_moire = ratio > MINE_LOSS_RATIO, moire > MOIRE_T
    flagged = (by_loss | by_moire).nonzero().flatten()
    print(f'{COUNT} crops scored, mean loss {loss.mean():.5f}: '
          f'{int(by_loss.sum())} over {MINE_LOSS_RATIO}x the mean, {int(by_moire.sum())} moire over {MOIRE_T}, '
          f'{len(flagged)} flagged ({100 * len(flagged) / COUNT:.1f}%)')

    # most severe first: the larger of the two scores against their thresholds
    severity = torch.maximum(ratio / MINE_LOSS_RATIO, moire / MOIRE_T)[flagged]
    shown = flagged[severity.argsort(descending=True)][:SHOW].tolist()
    rows = []
    for rank, i in enumerate(shown):
        f = f'{args.model}_{rank:03d}.png'
        im = Image.fromarray(cut[i].permute(1, 2, 0).numpy())
        im.resize((mod.PATCH * SCALE,) * 2, Image.NEAREST).save(OUT / f)
        why = ' + '.join(n for n, on in (('loss', by_loss[i]), ('moire', by_moire[i])) if on)
        rows.append((f, f'{rank}: loss x{ratio[i]:.2f}, moire {moire[i]:.2f} ({why})'))
    body = ''.join(f'<figure><img src="{f}"><figcaption>{html.escape(c)}</figcaption></figure>'
                   for f, c in rows)
    (OUT / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Mined crops</title>'
        '<style>body{background:#222;color:#ccc;font:12px sans-serif}'
        'figure{display:inline-block;margin:6px}img{image-rendering:pixelated;display:block}'
        f'figcaption{{max-width:{mod.PATCH * SCALE}px}}</style>' + body)
    print(f'{len(rows)} crops in {OUT}')


if __name__ == '__main__':
    main()
