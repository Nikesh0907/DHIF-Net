import os
import torch.utils.data as tud
import torch
import argparse
import numpy as np
import os
import sys

# Add CAVE folder to path to reuse Model and Utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CAVE')))
from Utils import *
from Utils import ensure_chw_numpy
from Model import HSI_Fusion
from Harvard_Dataset import harvard_dataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description="PyTorch Test for Harvard HSI Fusion")
parser.add_argument('--data_path', default='./Data/Test/', type=str, help='path of the testing data (expects HSI/ and RGB/ under this folder)')
parser.add_argument("--sizeI", default=512, type=int, help='the size of test crops')
parser.add_argument("--testset_num", default=12, type=int, help='total number of test samples')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
parser.add_argument("--seed", default=1, type=int, help='Random seed')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
parser.add_argument("--ckpt_path", default="", type=str, help="Path to a checkpoint .pth (overrides auto selection)")
parser.add_argument('--debug', action='store_true', help='Print per-sample debug stats (min/max/mean/mse)')
opt = parser.parse_args()
print(opt)


key = 'Test.txt'
file_path = os.path.join(opt.data_path, key)
if not os.path.exists(file_path):
    # fallback to repo Data/Test under HARVARD
    file_path = os.path.join(os.path.dirname(__file__), 'Data/Test', key)
    print(f"Test list not found in data_path. Falling back to {file_path}")
file_list = loadpath(file_path, shuffle=False)

# Create dataset and loader
dataset = harvard_dataset(opt, file_list, opt.data_path, istrain=False)
loader_test = tud.DataLoader(dataset, batch_size=opt.batch_size)

# Resolve checkpoint path (prefer explicit, else Harvard checkpoint dir)
ckpt_dir = os.path.join('.', 'Checkpoint', 'Harvard', f'f{opt.sf}', 'Model')
ckpt_path = None
if opt.ckpt_path:
    if not os.path.isfile(opt.ckpt_path):
        raise FileNotFoundError(f"--ckpt_path not found: {opt.ckpt_path}")
    ckpt_path = opt.ckpt_path
else:
    best_path = os.path.join(ckpt_dir, "model_best.pth")
    if os.path.isfile(best_path):
        ckpt_path = best_path
    elif os.path.isdir(ckpt_dir):
        # find last numeric checkpoint
        from Utils import findLastCheckpoint
        last_epoch = findLastCheckpoint(save_dir=ckpt_dir)
        if last_epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_{last_epoch:03d}.pth")
    if ckpt_path is None:
        # fallback legacy
        ckpt_path = os.path.join(ckpt_dir, 'model.pth')

print(f"Loading checkpoint: {ckpt_path}")

loaded = None
try:
    loaded = torch.load(ckpt_path, weights_only=False)
except TypeError:
    loaded = torch.load(ckpt_path)
except Exception as e:
    print(f"Primary load attempt failed: {e}\nTrying state_dict fallback...")


def _as_state_dict(obj):
    if isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            return obj['model']
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
        return obj
    return None


if isinstance(loaded, torch.nn.Module):
    model = loaded
else:
    model = HSI_Fusion(Ch=31, stages=4, sf=opt.sf)
    state = _as_state_dict(loaded)
    if state is None:
        raise RuntimeError("Unsupported checkpoint format: expected module or state_dict")
    if any(k.startswith('module.') for k in state.keys()):
        print("Stripping 'module.' prefixes from state_dict keys (DataParallel -> single GPU)")
        state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

model = model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

psnr_total = 0.0
sam_total = 0.0
ergas_total = 0.0
ssim_total = 0.0
k = 0
for j, (LR, RGB, HR) in enumerate(loader_test):
    with torch.no_grad():
        out = model(RGB.to(device), LR.to(device))
        result = out.clamp(min=0., max=1.)
    res_np = result.cpu().detach().numpy()
    hr_np = HR.numpy()
    if opt.debug:
        try:
            import numpy as _np
            res_flat = _np.asarray(res_np).ravel()
            hr_flat = _np.asarray(hr_np).ravel()
            mse = _np.mean((res_flat - hr_flat) ** 2)
            print(f'[DEBUG] sample {j}: res_min={res_flat.min():.6f} res_max={res_flat.max():.6f} hr_min={hr_flat.min():.6f} hr_max={hr_flat.max():.6f} mse={mse:.6e}')
        except Exception as _e:
            print('[DEBUG] could not compute debug stats:', _e)
    res_np = ensure_chw_numpy(np.squeeze(res_np))
    hr_np = ensure_chw_numpy(np.squeeze(hr_np))
    psnr = compare_psnr(hr_np, res_np, data_range=1.0)
    sam = compute_sam(hr_np, res_np)
    ergas = compute_ergas(hr_np, res_np, scale=opt.sf)
    ssim = compute_ssim(hr_np, res_np, data_range=1.0)
    psnr_total += psnr
    sam_total += sam
    ergas_total += ergas
    ssim_total += ssim
    k = k + 1

print(k)
print("Avg PSNR  = %.4f" % (psnr_total/k))
print("Avg SAM   = %.4f" % (sam_total/k))
print("Avg ERGAS = %.4f" % (ergas_total/k))
print("Avg SSIM  = %.4f" % (ssim_total/k))
