import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler

# Make CAVE's folder importable (re-use Model and utility functions)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CAVE')))
from Model import HSI_Fusion
from Utils import loadpath, dataparallel
from Harvard_Dataset import harvard_dataset

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def dataparallel(model, ngpus, gpu0=0):
    # reuse the same dataparallel helper as in CAVE/Utils but lightweight
    has_cuda = torch.cuda.is_available()
    if has_cuda and ngpus >= 1 and torch.cuda.device_count() >= gpu0 + ngpus:
        gpu_list = list(range(gpu0, gpu0 + ngpus))
        if ngpus > 1:
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
        return model
    else:
        print("[Warning] CUDA not available or insufficient GPUs; running on CPU.")
        return model.cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DHIF-Net on Harvard dataset")
    parser.add_argument('--data_path', required=True, help='Path to Harvard Data/Train folder (contains HSI/, RGB/, Train.txt)')
    parser.add_argument('--sizeI', default=96, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--trainset_num', default=20000, type=int)
    parser.add_argument('--sf', default=8, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--kernel_type', default='gaussian_blur', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=True)
    parser.add_argument('--resume_path', default='', type=str)
    opt = parser.parse_args()

    print('Random Seed: ', opt.seed)
    # Configure visible GPUs BEFORE any cuda calls (match CAVE behavior)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if opt.gpus is not None and opt.gpus > 0:
        try:
            ids = ",".join(str(i) for i in range(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ids
        except Exception:
            pass
    # quick runtime visibility logs (help debug GPU usage)
    print('CUDA_VISIBLE_DEVICES ->', os.environ.get('CUDA_VISIBLE_DEVICES'))

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    try:
        torch.cuda.manual_seed(opt.seed)
    except Exception:
        pass

# Enable cuDNN autotuner to select best algorithms for your HW (speeds up convolutions)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

    print(opt)

    # Model
    print('===> New Model')
    model = HSI_Fusion(Ch=31, stages=4, sf=opt.sf)
    print('===> Setting GPU')
    # Use the repository dataparallel helper (sets cuda and wraps DataParallel)
    model = dataparallel(model, opt.gpus)
    # report actual CUDA availability and device count after wrapper
    print('torch.cuda.is_available():', torch.cuda.is_available())
    try:
        print('torch.cuda.device_count():', torch.cuda.device_count())
    except Exception:
        pass

    # Initialize weights (same logic as original)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            try:
                nn.init.xavier_uniform_(layer.weight)
            except Exception:
                pass

    # Load file list
    file_path = os.path.join(opt.data_path, 'Train.txt')
    if not os.path.exists(file_path):
        print(f"Train list not found in data_path: {file_path}")
        raise SystemExit(1)
    file_list = loadpath(file_path)

    # Dataset uses on-the-fly loader
    dataset = harvard_dataset(opt, file_list, opt.data_path)
    # increase num_workers to speed up data loading when GPUs are available
    # choose a sensible default and cap it according to CPU count
    cpu_count = max(1, (os.cpu_count() or 4))
    default_workers = 8 if torch.cuda.is_available() else 2
    max_workers = max(1, min(default_workers, cpu_count - 1))
    num_workers = max(1, max_workers)
    # use pinned memory and persistent workers for faster host->device transfer
    dataloader_kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))
    loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, **dataloader_kwargs)
    print(f"DataLoader: num_workers={num_workers}, pin_memory={dataloader_kwargs['pin_memory']}, persistent_workers={dataloader_kwargs['persistent_workers']}")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95)

    start_epoch = 0
    # resume logic (simple)
    if opt.resume and opt.resume_path and os.path.exists(opt.resume_path):
        print(f"Resuming from: {opt.resume_path}")
        loaded = torch.load(opt.resume_path, map_location='cpu')
        if isinstance(loaded, dict) and 'epoch' in loaded:
            model.load_state_dict(loaded.get('model') or loaded.get('state_dict') or loaded, strict=False)
            if 'optimizer' in loaded:
                try:
                    optimizer.load_state_dict(loaded['optimizer'])
                except Exception:
                    pass
            start_epoch = int(loaded.get('epoch', 0)) + 1
        elif isinstance(loaded, dict):
            try:
                model.load_state_dict(loaded, strict=False)
            except Exception:
                pass
        elif isinstance(loaded, torch.nn.Module):
            try:
                model.load_state_dict(loaded.state_dict(), strict=False)
            except Exception:
                model = loaded

    print(f"Starting training from epoch {start_epoch} to {opt.epochs}")

    # AMP scaler when using CUDA
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        pbar = tqdm(enumerate(loader_train), total=len(loader_train), desc=f"Epoch {epoch+1}/{opt.epochs}")
        for i, (LR, RGB, HR) in pbar:
            device = next(model.parameters()).device
            LR = LR.to(device)
            RGB = RGB.to(device)
            HR = HR.to(device)

            # mixed precision forward/backward when CUDA available
            if use_amp:
                with autocast():
                    out = model(RGB, LR)
                    loss = criterion(out, HR)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                out = model(RGB, LR)
                loss = criterion(out, HR)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            avg_loss = epoch_loss / (i + 1)
            pbar.set_postfix(loss=f"{avg_loss:.6f}")

        scheduler.step()
        elapsed_time = time.time() - start_time
        print('epoch = %4d , avg_loss = %.10f , time = %4.2f s , lr = %.6e' % (
            epoch + 1, epoch_loss / max(1, len(loader_train)), elapsed_time, optimizer.param_groups[0]['lr']))

        # Save weights-only snapshot
        ckpt_dir = os.path.join('.', f'Checkpoint/Harvard/f{opt.sf}/Model')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_{epoch + 1:03d}.pth'))
