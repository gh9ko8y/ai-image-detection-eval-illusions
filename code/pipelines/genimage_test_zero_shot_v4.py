### GENEVAL_V4 ### TIFS 审稿 M9 修正版（原 genimage_test_zero_shot.py 保持不变）
"""GenImage 测试集零样本评估 v4。
修正：[M9] 三骨干除平衡准确率外，逐轮保存 acc_fake / acc_real（旧版只存平衡准确率，论文误标为 fake-accuracy）；
      [S12] 可选 --real-parquet：改用 GenImage 测试集自带的真图（nature）做真图侧，消除"真图来自 Tiny-GenImage"的来源混杂。
      JSON 键名 SD15 -> SD1.4（对应 stable_diffusion_v_1_4 分片）。
输出：runs/genimage_test_eval/result_v4.json
原说明：GenImage 测试集（第二数据集）零样本评估 —— parquet 适配版。
数据：genimage_parquet/data/*-00000-of-*.parquet（8 生成器，每片 3000 假图）。
真图：Tiny-GenImage 验证集 Real 300 张（同一 ImageNet 分布域）。
模型：三骨干 full ×3 轮 + UnivFD（ViT-L + full 头）。
输出 runs/genimage_test_eval/result.json"""
import argparse
import io
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

BASE = os.environ.get('DATA_ROOT', '<DATA_ROOT>')
PQDIR = BASE + '/genimage_parquet/data'
OUTDIR = BASE + '/runs/genimage_test_eval'
os.makedirs(OUTDIR, exist_ok=True)
DEV = 'cuda'
N_FAKE = 300
N_REAL = 300
GENS = ['ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD1.4', 'VQDM', 'Wukong']
PQNAME = {'ADM': 'ADM-00000-of-00004.parquet',
          'BigGAN': 'BigGAN-00000-of-00004.parquet',
          'GLIDE': 'glide-00000-of-00004.parquet',
          'Midjourney': 'Midjourney-00000-of-00004.parquet',
          'SD1.4': 'stable_diffusion_v_1_4-00000-of-00004.parquet',
          'VQDM': 'VQDM-00000-of-00004.parquet',
          'Wukong': 'wukong-00000-of-00004.parquet'}


def decode_parquet_side(path, n, rng):
    df = pd.read_parquet(path)
    take = rng.choice(len(df), min(n, len(df)), replace=False)
    arrs = []
    for i in take:
        rec = df.iloc[i]['image']
        raw = rec['bytes'] if isinstance(rec, dict) else rec
        im = Image.open(io.BytesIO(raw)).convert('RGB').resize(
            (224, 224), Image.BICUBIC)
        arrs.append(np.asarray(im, dtype=np.uint8))
    return np.stack(arrs)


def norm_with(x_u8, mean, std):
    x = x_u8.astype('float32') / 255.0
    x = x.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray((x - mean) / std))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-parquet', default=None,
                    help='GenImage test-set real (nature) parquet; default: Tiny-GenImage validation reals')
    ap.add_argument('--out', default=os.path.join(OUTDIR, 'result_v4.json'))
    args = ap.parse_args()
    rng = np.random.RandomState(42)
    if args.real_parquet:
        X_real = decode_parquet_side(args.real_parquet, N_REAL, rng)
        real_source = 'genimage_test:' + os.path.basename(args.real_parquet)
    else:
        X_real_all = np.load(BASE + '/data_npy/X_validation.npy', mmap_mode='r')
        meta = pd.read_csv(BASE + '/data_npy/validation_meta_named.csv')
        ridx = rng.choice(np.where(meta['gen_name'].values == 'Real')[0],
                          N_REAL, replace=False)
        X_real = np.asarray(X_real_all[ridx])
        real_source = "tiny_genimage_validation"
    print('real side:', X_real.shape, flush=True)

    fakes = {}
    for gen in GENS:
        fakes[gen] = decode_parquet_side(
            os.path.join(PQDIR, PQNAME[gen]), N_FAKE, rng)
        print('loaded fake', gen, fakes[gen].shape, flush=True)

    import bb_models
    import bb_clip_models

    configs = []
    configs.append(('resnet', bb_models.build_resnet18,
                    bb_models.RN18_MEAN, bb_models.RN18_STD,
                    [BASE + '/runs/round_01/detector.pth',
                     BASE + '/runs/round_02/detector.pth',
                     BASE + '/runs/round_03/detector.pth']))
    configs.append(('effnet', bb_models.build_efficientnet_b0,
                    bb_models.RN18_MEAN, bb_models.RN18_STD,
                    [BASE + '/runs/bb01/det_efficientnet_b0_full_r1.pth',
                     BASE + '/runs/bb01/det_efficientnet_b0_full_r2.pth',
                     BASE + '/runs/bb01/det_efficientnet_b0_full_r3.pth']))
    configs.append(('clip', bb_clip_models.build_clip,
                    bb_clip_models.CLIP_MEAN, bb_clip_models.CLIP_STD,
                    [BASE + '/runs/bb02/det_clip_vitb16_full_r1.pth',
                     BASE + '/runs/bb02/det_clip_vitb16_full_r2.pth',
                     BASE + '/runs/bb02/det_clip_vitb16_full_r3.pth']))

    result = {}
    for arch, builder, mean, std, ckpts in configs:
        model = builder().to(DEV).eval()
        per_gen = {}
        for gen, X_fake in fakes.items():
            accs, aurocs, afs, ars = [], [], [], []
            for ck in ckpts:
                model.load_state_dict(torch.load(ck, map_location=DEV,
                                                 weights_only=True))
                sf, sr = [], []
                with torch.no_grad():
                    for arr, is_real in [(X_fake, False), (X_real, True)]:
                        for i in range(0, len(arr), 200):
                            xb = norm_with(arr[i:i + 200], mean, std).to(DEV)
                            pp = F.softmax(model(xb), 1)[:, 1].cpu().numpy()
                            if is_real:
                                sr.append(pp)
                            else:
                                sf.append(pp)
                sf = np.concatenate(sf)
                sr = np.concatenate(sr)
                afs.append(float((sf >= 0.5).mean() * 100))
                ars.append(float((sr < 0.5).mean() * 100))
                accs.append((afs[-1] + ars[-1]) / 2)
                aurocs.append(roc_auc_score(
                    np.r_[np.ones(len(sf)), np.zeros(len(sr))],
                    np.r_[sf, sr]) * 100)
            per_gen[gen] = {'acc_balanced': float(np.mean(accs)),
                            'acc_balanced_range': float(max(accs) - min(accs)),
                            'acc_fake': float(np.mean(afs)), 'acc_fake_runs': afs,
                            'acc_real': float(np.mean(ars)), 'acc_real_runs': ars,
                            'auroc': float(np.mean(aurocs)), 'auroc_runs': aurocs}
            print(arch, gen, per_gen[gen], flush=True)
        result[arch] = per_gen
        del model
        torch.cuda.empty_cache()

    # UnivFD 特征级（重用已提取的 feat_validation + 现场提 GenImage 特征）
    import open_clip
    import pickle
    m, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', force_quick_gelu=True)
    vis = m.visual.to(DEV).eval()

    @torch.no_grad()
    def clip_feats(arr):
        fs = []
        for i in range(0, len(arr), 100):
            xb = norm_with(arr[i:i + 100], bb_clip_models.CLIP_MEAN,
                           bb_clip_models.CLIP_STD).to(DEV)
            with torch.autocast('cuda', dtype=torch.float16):
                fs.append(vis(xb).float().cpu().numpy())
        return np.concatenate(fs)

    with open(FEATDIR_PATH := BASE + '/runs/univfd/heads/full.pkl',
              'rb') as fh:
        clf = pickle.load(fh)
    s_real = clf.predict_proba(clip_feats(X_real))[:, 1]
    entry_u = {}
    for gen, X_fake in fakes.items():
        s_fake = clf.predict_proba(clip_feats(X_fake))[:, 1]
        entry_u[gen] = {
            'acc_balanced': float(((s_fake >= 0.5).mean() * 100 + (s_real < 0.5).mean() * 100) / 2),
            'acc_fake': float((s_fake >= 0.5).mean() * 100),
            'acc_real': float((s_real < 0.5).mean() * 100),
            'auroc': float(roc_auc_score(
                np.r_[np.ones(len(s_fake)), np.zeros(len(s_real))],
                np.r_[s_fake, s_real]) * 100)}
        print('univfd', gen, entry_u[gen], flush=True)
    result['univfd'] = entry_u
    result['_meta'] = {'n_fake': N_FAKE, 'n_real': N_REAL, 'real_source': real_source}

    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=1)
    print('GENIMAGE_EVAL_V4_DONE', flush=True)


main()
