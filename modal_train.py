"""Run a train_*.py on a Modal GPU (https://modal.com). Everything lives on the
Modal volume `starfish`: the data under /data, and a copy of training/ under
/training, which is where the checkpoint and the export are written, so a
run that is stopped or preempted resumes from its last 10000-step save.

Once:
    pip install modal && modal setup
    modal volume create starfish
    modal volume put starfish ../../demosaic/div2k.npy /data/div2k.npy
    modal volume put starfish ../../demosaic/div2k_index.npy /data/div2k_index.npy

Per run (from training/; --detach keeps it running after the terminal closes):
    modal run --detach modal_train.py --script train_xtrans.py
    modal run --detach modal_train.py --script "finetune.py train_xtrans"
A fine-tune needs the trained checkpoint on the volume first:
    modal volume put starfish xtrans.pt /training/xtrans.pt

Afterwards:
    modal volume get starfish /training/xtrans.pt .
    modal volume get starfish /training/xtrans_wgpu .
    (a fine-tune: /training/xtrans_ft.pt and /training/xtrans_ft_wgpu)
"""
import shutil
import subprocess
from pathlib import Path

import modal

GPU = 'H100'            # or 'L40S', 'A100-80GB': any CUDA 12.8 GPU
HOURS = 24                   # Modal's longest timeout; a stopped run resumes

app = modal.App('starfish-training')
volume = modal.Volume.from_name('starfish')
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('gcc')      # torch.compile's Triton builds a small C launcher
    .pip_install('torch==2.11.0', 'numpy==2.5.2', 'pillow==12.3.0',
                 index_url='https://download.pytorch.org/whl/cu128',
                 extra_index_url='https://pypi.org/simple')
    .add_local_dir(Path(__file__).parent, '/root/src',
                   ignore=['.venv', 'data', '__pycache__', '*.pt', '*_wgpu'])
)


@app.function(gpu=GPU, image=image, volumes={'/vol': volume}, cpu=8, memory=32768,
              timeout=HOURS * 3600, retries=modal.Retries(max_retries=3), scaledown_window=5)
def train(script: str):
    # The data to local disk: the crops read it at random, which a network
    # volume does slowly.
    for name in ('div2k.npy', 'div2k_index.npy'):
        shutil.copy(f'/vol/data/{name}', f'/tmp/{name}')
    # The code next to the checkpoint, as on the desktop (HERE in common.py)
    run = Path('/vol/training')
    run.mkdir(exist_ok=True)
    for src in Path('/root/src').glob('*.py'):
        shutil.copy(src, run)
    try:
        # -u: unbuffered, so progress lines show as they are printed, not in 8 KB bursts
        # the script and its arguments, e.g. "finetune.py train_xtrans"
        subprocess.run(['python', '-u', *script.split(), '--data', '/tmp/div2k.npy'], cwd=run, check=True)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(script: str = 'train_xtrans.py'):
    train.remote(script)
