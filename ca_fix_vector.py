"""Writes ca_fix.bin, the test vector of the app's CA pass (src/processing/ca_fix.rs,
matches_pytorch): a Kodak image through the sensor with TEST_CA,
common.ca_correct's output on it and the same mosaic without CA. All are written unbalanced, with the
white-balance gains, as the app runs the pass on the frame as decoded.

    python ca_fix_vector.py [xtrans|bayer] [kodim19]
    STARFISH_TEST_CA_FIX_BIN=training/ca_fix.bin cargo test --release matches_pytorch -- --ignored --nocapture
"""
import struct
import sys

import numpy as np
import torch
from PIL import Image

from common import (BAYER, CAM_FROM_SRGB_BANK, HERE, TEST_CA, TEST_WB, XTRANS, ca_correct, cfa_masks, radial_ca,
                    sensor, srgb_to_linear)

cfa = (XTRANS if (sys.argv[1:] or ['xtrans'])[0] == 'xtrans' else BAYER).cuda()
im = np.array(Image.open(HERE.parent.parent / 'demosaic' / 'images' / ((sys.argv[2:] or ['kodim19'])[0] + '.png')).convert('RGB'))
h, w = im.shape[0] // 6 * 6, im.shape[1] // 6 * 6
lin = srgb_to_linear(torch.from_numpy(im[:h, :w]).cuda().permute(2, 0, 1)[None] / 255)
one = lambda v: torch.full((1, 1, 1, 1), float(v), device='cuda')
cam = CAM_FROM_SRGB_BANK[:1].cuda()
gain = torch.tensor([TEST_WB[0], 1.0, TEST_WB[1]], device='cuda').view(1, 3, 1, 1)
field = radial_ca(h, w, *TEST_CA, 'cuda')
m = sensor(lin, cfa, cam, torch.linalg.inv(cam), gain, one(1), one(0), one(0), one(0), one(0), field)[0]
fixed = ca_correct(m, cfa, field, clamp=False)
# the same photograph without CA: what the correction is after
clean = sensor(lin, cfa, cam, torch.linalg.inv(cam), gain, one(1), one(0), one(0), one(0), one(0), None)[0]
site_gain = (gain * cfa_masks(cfa, h, w).cuda()).sum(1, keepdim=True)
half = np.hypot(h, w) / 2
with open(HERE / 'ca_fix.bin', 'wb') as f:
    f.write(struct.pack('<III', h, w, cfa.shape[0]))
    f.write(struct.pack('<ff', TEST_CA[0] / half, TEST_CA[1] / half))
    f.write(struct.pack('<fff', TEST_WB[0], 1.0, TEST_WB[1]))
    f.write(cfa.flatten().cpu().numpy().astype('<u4').tobytes())
    for t in (m / site_gain, fixed / site_gain, clean / site_gain):
        f.write(t.float().cpu().numpy().astype('<f4').tobytes())
