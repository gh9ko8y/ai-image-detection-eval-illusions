### GEN_v2 ### strip marker lines
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
print('cuda avail:', torch.cuda.is_available(), flush=True)
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_single_file(
    os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'hf_models/sdxl-base/sd_xl_base_1.0.safetensors'),
    config=os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'hf_models/sdxl-base'),
    torch_dtype=torch.float16, local_files_only=True)
pipe = pipe.to('cuda')
p = next(pipe.unet.conv_in.parameters())
print('unet on', p.device, p.dtype, flush=True)

TEMPLATES = [
    'a photo of a {}', 'a high quality photo of a {}',
    'a close-up photo of a {}', 'a bright photo of a {}',
    'a cropped photo of a {}', 'a tattoo photo of a {}',
    'a photo of one {}', 'a photo of the {}',
    'a good photo of a {}', 'a photo of many {}',
]
from torchvision.models import ResNet18_Weights
classes = [c.split(', ')[0].replace('_', ' ') for c in
           ResNet18_Weights.IMAGENET1K_V1.meta['categories']]
TOTAL = 500
plan = []
for i in range(TOTAL):
    cls = classes[i % len(classes)]
    tmpl = TEMPLATES[(i // len(classes)) % len(TEMPLATES)]
    plan.append(tmpl.format(cls))
rng = np.random.RandomState(42)
plan = [plan[i] for i in rng.permutation(len(plan))]
print('plan ready', len(plan), flush=True)

OUTDIR = os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'runs/sdxl_zeroshot/gen')
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(OUTDIR)
done = len([n for n in os.listdir('.') if n.endswith('.png')])
print('resume from', done, flush=True)
while done < TOTAL:
    prompt_text = plan[done]
    image = pipe(prompt=prompt_text, num_inference_steps=30,
                 guidance_scale=5.0).images[0]
    name = 'img_' + str(done).zfill(5) + '.png'
    image.resize((512, 512)).save(name)
    done += 1
    if done % 20 == 0:
        print('generated', done, 'of', TOTAL, flush=True)
with open('done.txt', 'w') as handle:
    handle.write('ok')
print('GENERATION_COMPLETE', flush=True)
