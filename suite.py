"""The artifact test suite: run trained models on fixed test images and print
PSNR per category, one table per CFA, so a change shows up as a category and
not only as a fraction of a dB of held-out PSNR.

    python suite.py                                  # every train_*.py whose checkpoint exists
    python suite.py train_bayer train_bayer_lite    # just these

A model with a fine-tuned copy (<name><FINETUNE_SUFFIX>.pt, finetune.py) gets a
second column, "<name> ftl", for before/after. (The 2026-10-17 <name>_ft.pt
files are of the older architecture and no longer load.)

Writes suite/<model>/<case>.png (sRGB) to look at. Categories:

    held-out   training's held-out crops (eval_set, its seed), loss border BORDER
    kodak      IMAGES/kodim*.png
    zone       a grey zone plate up to Nyquist
    lines      gratings of thin saturated lines at 16 angles, on grey
    edges      edges between saturated colours at 16 angles
    clipped    blown highlights, single-channel clipping and ramps into clipping,
               next to saturated colours
    dark       flat colour patches 3 EV under, strong noise
    pattern    flat grey at PATTERN_LEVELS through the noisy sensor PATTERN_DRAWS
               times: the output's fixed error per CFA position (its mean
               over the draws, per position), as -20 log10(its sd / the
               level), the mean over levels (higher = less pattern)
    symmetry   net(g(m)) with g undone against net(m), for the 7 flips and
               rotations g that map the CFA onto itself after a shift: how far
               the model is from treating every orientation alike (inf = exact)
    push N     kodak shot N EV under (exposure 2^-N, NOISE) and pushed back
               up, as a user lifting shadows sees it: PSNR after the loss's
               old display curve (common.tone, gamma 1/2.2), PUSHES
    push N chroma   the same of the colour alone: tone(C) - tone(luma)
    lift N     shadows() shot at its exposure (NOISE) and shown N EV brighter
               (x 2^N, highlights clip): a user lifting an image's shadows, LIFTS.
               Synthetic, so the truth has no 8-bit steps to lift into bands
    CA fixed   kodak with a radial lateral CA (common.radial_ca, TEST_CA: red
               and blue shifted by 2 / -2.5 px at the corner), taken out of the
               mosaic by common.ca_correct with the true field, as the app does,
               against the image without it

Every category but held-out, dark, pattern and push runs clean and noisy
(NOISE). PSNR is in linear sRGB clamped to [0, 1] (push: after tone),
BORDER px per side dropped (the networks
pad with zeros, which reach no further than that), run in fp32 with TF32 off.
"""
import importlib
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from finetune import FINETUNE_SUFFIX
from common import (CAM_FROM_SRGB_BANK, DATA, HERE, LUMA_SRGB, TEST_CA, TEST_WB, apply_ccm, ca_correct,
                    eval_set, evaluate, linear_to_srgb, psnr, radial_ca, sensor, srgb_to_linear, tone)

IMAGES = HERE.parent.parent / 'demosaic' / 'images'   # kodim*.png
OUT = HERE / 'suite'
BORDER = 36                  # px per side dropped: at least every model's reach
SIZE = 384                   # synthetic images, a multiple of 6 and of the 4x4 grid
NOISE = (1e-3, 1e-6)         # a, b of the noisy runs (the export's test vector's)
DARK = (1 / 8, 4e-3, 4e-5)   # exposure, a, b of the dark patches
SEED = 0
SAVED_KODAK = 4              # kodak images written as PNG (all are scored)
PATTERN_LEVELS = (0.005, 0.02, 0.08)   # linear grey of the pattern category
PATTERN_DRAWS = 32
PUSHES = (3, 5)              # EV under of the push categories
LIFTS = (3, 5)               # EV of display gain of the lift categories
SAVED_PUSH = 'kodim01'       # the kodak image written as PNG for the push categories
SHADOW_EV = (-10, -6)        # the shadow image's background, darkest and brightest (white = 0)
SHADOW_TEXTURE = 0.15        # its texture's sd, of the level

SAT_COLOURS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]


# ------------------------------------------------------------- test images
# All linear sRGB, 1,3,H,W on the GPU.

def grid():
    """(y, x) px of a SIZE x SIZE image and, per 4x4 tile, its index and the
    px relative to its centre."""
    t = SIZE // 4
    y, x = torch.meshgrid(torch.arange(SIZE, device='cuda', dtype=torch.float32),
                          torch.arange(SIZE, device='cuda', dtype=torch.float32), indexing='ij')
    k = (y // t) * 4 + x // t
    return y, x, k, y % t - t / 2 + 0.5, x % t - t / 2 + 0.5


def zone_plate():
    y, x, *_ = grid()
    r2 = (y - SIZE / 2 + 0.5) ** 2 + (x - SIZE / 2 + 0.5) ** 2
    v = 0.5 + 0.4 * torch.cos(math.pi * r2 / SIZE)          # 0.5 cycles/px at the edge midpoints
    return srgb_to_linear(v)[None, None].expand(1, 3, -1, -1).contiguous()


def lines():
    _, _, k, ty, tx = grid()
    th = k * math.pi / 16
    d = (tx * torch.sin(th) - ty * torch.cos(th)).abs() % 7    # parallel lines 7 px apart
    cover = (1.25 - d.minimum(7 - d)).clamp(0, 1)             # ~1.5 px wide, antialiased
    col = torch.tensor(SAT_COLOURS, device='cuda')[k.long() % 6].permute(2, 0, 1)
    grey = srgb_to_linear(torch.tensor(0.5))
    return (grey + cover * (col * 0.8 - grey))[None]


def edges():
    _, _, k, ty, tx = grid()
    th = k * math.pi / 16 + 0.1
    cover = (0.5 + tx * torch.sin(th) - ty * torch.cos(th)).clamp(0, 1)
    pal = torch.tensor(SAT_COLOURS, device='cuda') * 0.8
    a = pal[k.long() % 6].permute(2, 0, 1)
    b = pal[(k.long() + 2 + k.long() // 6) % 6].permute(2, 0, 1)
    return (a + cover * (b - a))[None]


def clipped():
    y, x, k, ty, tx = grid()
    t = SIZE // 4
    pal = torch.tensor(SAT_COLOURS, device='cuda')
    bg = pal[k.long() % 6].permute(2, 0, 1) * 0.6
    r = (ty ** 2 + tx ** 2).sqrt()
    disc = (t / 4 - r).clamp(0, 1)                             # blown white-ish disc
    hot = torch.tensor([1.0, 0.9, 0.75], device='cuda').view(3, 1, 1) * 4
    one = pal[(k.long() + 3) % 6].permute(2, 0, 1) * 2 + 0.05  # one or two channels clip
    ramp = ((tx + t / 2) / t * 3)                               # 0 .. 3: into clipping
    rows = (k // 4).long()
    img = torch.where(rows % 2 == 0, bg + disc * (hot - bg), bg + disc * (one - bg))
    img = torch.where(rows == 3, pal[k.long() % 6].permute(2, 0, 1) * ramp + 0.02, img)
    return img[None]


def patches():
    _, _, k, *_ = grid()
    rng = np.random.default_rng(SEED)
    cols = torch.tensor(rng.uniform(0.1, 0.9, (16, 3)), device='cuda', dtype=torch.float32)
    return srgb_to_linear(cols[k.long()].permute(2, 0, 1))[None]


def smooth_noise(g, h, w, scale):
    """A fixed random field blurred to `scale` px, sd 1."""
    r = int(3 * scale)
    k = torch.exp(-torch.arange(-r, r + 1, device='cuda', dtype=torch.float32) ** 2 / (2 * scale ** 2))
    k = k / k.sum()
    x = torch.randn(1, 1, h + 2 * r, w + 2 * r, generator=g, device='cuda')
    x = F.conv2d(F.conv2d(x, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    return x / x.std()


def shadows():
    """A dark 2 SIZE square: four bands of SIZE / 2 rows over a background
    that falls from 2^SHADOW_EV[1] at the left to 2^SHADOW_EV[0] at the right,
    with a faint fixed texture (smooth noise at 1.5, 3 and 6 px, sd
    SHADOW_TEXTURE of the level): texture alone; slightly saturated colour
    patches; faint lines and edges (x 1.5 and x 2 the level); bright discs,
    some saturated, whose surroundings lifting shows."""
    n = 2 * SIZE
    g = torch.Generator(device='cuda').manual_seed(SEED)
    y, x = torch.meshgrid(torch.arange(n, device='cuda', dtype=torch.float32),
                          torch.arange(n, device='cuda', dtype=torch.float32), indexing='ij')
    lo, hi = SHADOW_EV
    level = 2 ** (hi + (lo - hi) * x / (n - 1))
    texture = sum(smooth_noise(g, n, n, sc) for sc in (1.5, 3, 6)) / math.sqrt(3)
    img = (level * (1 + SHADOW_TEXTURE * texture.clamp(-2.5, 2.5)))[0].expand(3, -1, -1).clone()
    band, by = (y // (n // 4)).long(), y % (n // 4)
    pal = torch.tensor(SAT_COLOURS, device='cuda')
    # band 1: colour patches, 48 px squares: a saturated colour plus grey, 3:1
    # (or 2:1) per channel, at the level's luma
    k = ((x // 48) + (y // 48) * 3).long() % 6
    tint = 0.5 + pal[k].permute(2, 0, 1)
    tint = tint / (tint * LUMA_SRGB.cuda().view(3, 1, 1)).sum(0, keepdim=True)
    img = torch.where(band == 1, img * tint, img)
    # band 2: lines (~1.5 px, x 1.5) at 4 angles in its top half, edges (x 2) below
    th = (x // 96) * math.pi / 8
    d = (x * torch.sin(th) - y * torch.cos(th)).abs() % 12
    line = 1 + 0.5 * (1.25 - d.minimum(12 - d)).clamp(0, 1)
    edge = 1 + (0.5 + (x % 96 - 48) * torch.cos(th) + (by - n / 8 - 24) * torch.sin(th)).clamp(0, 1)
    img = torch.where((band == 2) & (by < n / 8), img * line, img)
    img = torch.where((band == 2) & (by >= n / 8), img * edge, img)
    # band 3: bright discs, radius 14 px, every 96 px
    r = ((x % 96 - 48) ** 2 + (by - n / 8) ** 2).sqrt()
    disc = (14 - r).clamp(0, 1)
    bright = torch.where(((x // 96).long() % 2 == 0), torch.ones_like(x), torch.zeros_like(x))
    col = torch.where(bright.bool(), torch.ones(3, 1, 1, device='cuda') * 0.8, pal[(x // 192).long() % 6].permute(2, 0, 1) * 0.8)
    img = torch.where(band == 3, img + disc * (col - img), img)
    return img[None]


def kodak():
    for f in sorted(IMAGES.glob('kodim*.png')):
        im = np.array(Image.open(f).convert('RGB'))
        h, w = im.shape[0] // 6 * 6, im.shape[1] // 6 * 6
        yield f.stem, srgb_to_linear(torch.from_numpy(im[:h, :w]).cuda().permute(2, 0, 1)[None] / 255)


# ------------------------------------------------------------------ running

def run(net, cfa, lin, e=1.0, a=0.0, b=0.0, ca=None):
    """lin through the fixed sensor (with the lateral CA field `ca`, taken out
    of the mosaic again by ca_correct) and net -> (out, target), linear sRGB."""
    dev = lin.device
    one = lambda v: torch.full((1, 1, 1, 1), float(v), device=dev)
    cam = CAM_FROM_SRGB_BANK[:1].to(dev)
    gain = torch.tensor([TEST_WB[0], 1.0, TEST_WB[1]], device=dev).view(1, 3, 1, 1)
    torch.manual_seed(SEED)
    noisy, sigma, target, srgb_from_cam = sensor(
        lin, cfa, cam, torch.linalg.inv(cam), gain, one(e), one(a), one(b), one(a), one(b), ca)
    if ca is not None:
        noisy = ca_correct(noisy, cfa, ca)
    out = net(noisy, sigma)
    return apply_ccm(out, srgb_from_cam), apply_ccm(target, srgb_from_cam), (noisy, sigma), srgb_from_cam


def score(out, target):
    c = lambda t: t.clamp(0, 1)[..., BORDER:-BORDER, BORDER:-BORDER]
    return psnr(c(out), c(target)).item()


def save(out, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    px = (linear_to_srgb(out[0]).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    Image.fromarray(px).save(path)


DIHEDRAL = [  # (name, g, g^-1) on the last two dims; square images only
    ('rot90', lambda t: t.rot90(1, (-2, -1)), lambda t: t.rot90(-1, (-2, -1))),
    ('rot180', lambda t: t.rot90(2, (-2, -1)), lambda t: t.rot90(2, (-2, -1))),
    ('rot270', lambda t: t.rot90(3, (-2, -1)), lambda t: t.rot90(-3, (-2, -1))),
    ('flip-x', lambda t: t.flip(-1), lambda t: t.flip(-1)),
    ('flip-y', lambda t: t.flip(-2), lambda t: t.flip(-2)),
    ('transpose', lambda t: t.transpose(-2, -1), lambda t: t.transpose(-2, -1)),
    ('anti-transpose', lambda t: t.transpose(-2, -1).rot90(2, (-2, -1)),
     lambda t: t.transpose(-2, -1).rot90(2, (-2, -1))),
]


def pattern(net, cfa):
    """The pattern category: see the docstring at the top."""
    period = cfa.shape[0]
    scores = []
    for level in PATTERN_LEVELS:
        n = 8                                       # draws per batch
        lin = torch.full((n, 3, SIZE, SIZE), level, device='cuda')
        one = lambda v: torch.full((n, 1, 1, 1), float(v), device='cuda')
        cam = CAM_FROM_SRGB_BANK[:1].cuda().expand(n, 3, 3)
        gain = torch.tensor([TEST_WB[0], 1.0, TEST_WB[1]], device='cuda').view(1, 3, 1, 1).expand(n, 3, 1, 1)
        a, b = NOISE
        torch.manual_seed(SEED)
        errors = []
        for _ in range(PATTERN_DRAWS // n):         # a new noise draw per patch
            noisy, sigma, target, srgb_from_cam = sensor(
                lin, cfa, cam, torch.linalg.inv(cam), gain, one(1), one(a), one(b), one(a), one(b))
            out = apply_ccm(net(noisy, sigma), srgb_from_cam) - apply_ccm(target, srgb_from_cam)
            errors.append(out[..., BORDER:-BORDER, BORDER:-BORDER])
        fixed = torch.cat(errors).mean(0)                                   # 3, h, w
        ph = fixed.unflatten(-1, (-1, period)).unflatten(-3, (-1, period)).mean((-4, -2))
        ph = ph - ph.mean((-2, -1), keepdim=True)
        scores.append(-20 * math.log10(ph.std().item() / level))
    return float(np.mean(scores))


def pushed(net, cfa, out_dir):
    """The push categories: see the docstring at the top."""
    dev = 'cuda'
    one = lambda v: torch.full((1, 1, 1, 1), float(v), device=dev)
    cam = CAM_FROM_SRGB_BANK[:1].to(dev)
    gain = torch.tensor([TEST_WB[0], 1.0, TEST_WB[1]], device=dev).view(1, 3, 1, 1)
    a, b = NOISE
    view = lambda z: tone(z.clamp(0, 1))[..., BORDER:-BORDER, BORDER:-BORDER]
    luma = lambda z: (z * LUMA_SRGB.to(z).view(1, 3, 1, 1)).sum(1, keepdim=True)
    chroma = lambda z: view(z) - view(luma(z))
    r = {}
    # (name, exposure, display gain)
    cases = ([(f'push {n}', 2 ** -n, 1, kodak) for n in PUSHES] +
             [(f'lift {n}', 1, 2 ** n, lambda: [('shadows', shadows())]) for n in LIFTS])
    for key, e, shown, images in cases:
        for stem, lin in images():
            torch.manual_seed(SEED)
            noisy, sigma, target, srgb_from_cam = sensor(
                lin, cfa, cam, torch.linalg.inv(cam), gain, one(e), one(a), one(b), one(a), one(b))
            o = apply_ccm(net(noisy, sigma), srgb_from_cam) * shown
            t = apply_ccm(target, srgb_from_cam) * shown
            r.setdefault(key, []).append(psnr(view(o), view(t)).item())
            r.setdefault(f'{key} chroma', []).append(psnr(chroma(o), chroma(t)).item())
            if stem in (SAVED_PUSH, 'shadows'):
                save(o, out_dir / f'{stem}_{key.replace(" ", "")}.png')
    return r


def symmetry(net, cfa, lin):
    """Min PSNR over DIHEDRAL of the transformed run against the plain one."""
    out, _, inputs, srgb_from_cam = run(net, cfa, lin, 1.0, *NOISE)
    n, p = lin.shape[-1], cfa.shape[0]
    colours = cfa.repeat(n // p, n // p)
    worst = math.inf
    for name, g, gi in DIHEDRAL:
        moved = g(colours)
        shifts = [(sy, sx) for sy in range(p) for sx in range(p)
                  if torch.equal(moved.roll((sy, sx), (0, 1)), colours)]
        assert shifts, f'{name} does not map the CFA onto itself'
        s = shifts[0]
        o = net(*(g(t).roll(s, (-2, -1)) for t in inputs))
        back = apply_ccm(gi(o.roll((-s[0], -s[1]), (-2, -1))), srgb_from_cam)
        worst = min(worst, score(back, out))
    return worst


@torch.no_grad()
def suite(name, suffix=''):
    mod = importlib.import_module(name)
    ckpt = mod.CKPT.with_name(mod.CKPT.stem + suffix + '.pt')
    if not ckpt.exists():
        return None
    cfa = mod.CFA.cuda()
    net = mod.build().cuda().eval()
    net.load_state_dict(torch.load(ckpt, map_location='cuda')['model'])
    out_dir = OUT / (name.removeprefix('train_') + suffix)
    r = {}

    _, r['held-out'] = evaluate(net, eval_set(DATA, 192, cfa, 'cuda'), BORDER)

    def case(key, lin, e=1.0, a=0.0, b=0.0, file=None, ca=None):
        out, target, *_ = run(net, cfa, lin, e, a, b, ca)
        r.setdefault(key, []).append(score(out, target))
        if file:
            save(out, out_dir / f'{file}.png')

    for i, (stem, lin) in enumerate(kodak()):
        case('kodak', lin, file=stem if i < SAVED_KODAK else None)
        case('kodak noisy', lin, 1.0, *NOISE, file=f'{stem}_noisy' if i < SAVED_KODAK else None)
        field = radial_ca(*lin.shape[-2:], *TEST_CA, 'cuda')
        case('CA fixed', lin, file=f'{stem}_ca_fixed' if i < SAVED_KODAK else None, ca=field)
        case('CA fixed noisy', lin, 1.0, *NOISE, ca=field)
    for key, make in (('zone', zone_plate), ('lines', lines), ('edges', edges), ('clipped', clipped)):
        lin = make()
        case(key, lin, file=key)
        case(f'{key} noisy', lin, 1.0, *NOISE, file=f'{key}_noisy')
    case('dark', patches(), *DARK, file='dark')
    r['pattern'] = pattern(net, cfa)
    r.update(pushed(net, cfa, out_dir))
    sq = next(kodak())[1]
    r['symmetry'] = symmetry(net, cfa, sq[..., :min(sq.shape[-2:]), :min(sq.shape[-2:])])
    return {k: float(np.mean(v)) for k, v in r.items()}, cfa.shape[0]


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    names = sys.argv[1:] or sorted(f.stem for f in HERE.glob('train_*.py'))
    tables = {}
    for name in names:
        for suffix in ('', FINETUNE_SUFFIX):
            res = suite(name, suffix)
            if res is None:
                if not suffix:
                    print(f'{name}: no checkpoint, skipped')
                continue
            scores, period = res
            column = name.removeprefix('train_') + suffix.replace('_', ' ')
            tables.setdefault('Bayer' if period == 2 else 'X-Trans', {})[column] = scores
            print(f'{column}: done')
    for cfa, models in tables.items():
        keys = list(next(iter(models.values())))
        print(f'\n{cfa} (PSNR dB)')
        print(f'{"":20}' + ''.join(f'{m:>15}' for m in models))
        for k in keys:
            print(f'{k:20}' + ''.join(f'{models[m][k]:15.2f}' for m in models))


if __name__ == '__main__':
    main()
