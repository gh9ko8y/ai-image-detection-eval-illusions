import json, os, time

os.environ['HF_HOME'] = '$DATA_ROOT/hf_probe_tmp'
import torch
from diffusers import FluxPipeline

SNAP = ('$DATA_ROOT/'
        'AI-ModelScope--FLUX.1-schnell/snapshots/master')
pipe = FluxPipeline.from_pretrained(SNAP, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

OUT = '$DATA_ROOT/flux_schnell_500_v1'
os.makedirs(OUT, exist_ok=True)
prompts = json.load(open(
    '$DATA_ROOT/prompts_500_v1_sdxl_aligned.json'))
assert len(prompts) == 500

import glob
done = len(glob.glob(OUT + '/img_*.png'))
print('resume from', done, flush=True)
t0 = time.time()
for idx in range(done, 500):
    img = pipe(prompt=prompts[idx], num_inference_steps=4,
               guidance_scale=0.0, height=1024, width=1024,
               generator=torch.Generator().manual_seed(42 + idx)).images[0]
    img.resize((512, 512)).save(OUT + ('/img_%05d.png' % idx))
    if idx % 50 == 0:
        el = time.time() - t0
        print('idx=%d elapsed=%.0fs eta=%.0fmin' % (
            idx, el, (499 - idx) * el / max(idx - done + 1, 1) / 60),
            flush=True)
print('GEN_500_V1_DONE total=%.0fs' % (time.time() - t0), flush=True)

from PIL import Image
import numpy as np
bad = []
for idx in range(500):
    try:
        im = Image.open(OUT + ('/img_%05d.png' % idx)).convert('RGB')
        a = np.asarray(im, dtype=np.float32)
        assert im.size == (512, 512) and np.isfinite(a).all() \
            and 0 < a.mean() < 255
    except Exception as e:
        bad.append((idx, str(e)[:40]))
print('CHECK_500_V1 bad=%d %s' % (len(bad), bad[:5]), flush=True)
assert not bad
print('ALL_500_V1_VERIFIED', flush=True)
