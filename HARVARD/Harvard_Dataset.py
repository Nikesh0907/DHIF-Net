import os
import random
import scipy.io as sio
import numpy as np
import torch
import torch.utils.data as tud

# Reuse helper functions from CAVE by adding its folder to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CAVE')))
from Utils import para_setting


class harvard_dataset(tud.Dataset):
    """Dataset that loads Harvard .mat files on the fly and returns patches.

    Expects folder layout:
      data_path/HSI/<id>.mat  (variable 'hsi' or 'ref')
      data_path/RGB/<id>.mat  (variable 'rgb')
      data_path/Train.txt (list of ids) passed separately
    """
    def __init__(self, opt, file_list, data_path, istrain=True):
        super(harvard_dataset, self).__init__()
        self.path = data_path
        self.istrain = istrain
        self.factor = opt.sf
        self.file_list = file_list
        self.file_num = len(file_list)
        if istrain:
            self.num = opt.trainset_num
            self.sizeI = opt.sizeI
        else:
            self.num = opt.testset_num
            # for testing, you probably want full image-sized crops — keep default
            self.sizeI = 512
        # simple per-worker cache to avoid repeated slow .mat reads
        # Each worker process will get its own copy after fork which is desirable.
        self._cache = {}
        # Precompute FFT of blur kernel once (numpy complex array) for given patch size
        sz = [self.sizeI, self.sizeI]
        try:
            fft_B, fft_BT = para_setting(getattr(opt, 'kernel_type', 'gaussian_blur'), self.factor, sz, sigma=2.0)
            # keep numpy complex array for fast numpy.fft usage in workers
            self._fft_B_np = fft_B
        except Exception:
            # fallback: compute on the fly later
            self._fft_B_np = None
        # downsample offset used when taking samples after blur
        self._ds_offset = int(self.factor // 2) - 1

    def H_z(self, z, factor, fft_B):
        # z expected as torch tensor [1, C, H, W]
        device = z.device
        if len(z.shape) == 3:
            ch, h, w = z.shape
            f = torch.fft.fft2(z, dim=(-2, -1))
            fft_B_c = torch.complex(fft_B[:, :, 0].to(device), fft_B[:, :, 1].to(device))
            M = f * fft_B_c
            Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
            x = Hz[:, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            f = torch.fft.fft2(z, dim=(-2, -1))
            fft_B_c = torch.complex(fft_B[:, :, 0].to(device), fft_B[:, :, 1].to(device)).unsqueeze(0).unsqueeze(0)
            M = f * fft_B_c
            Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
            x = Hz[:, :, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        return x

    def __getitem__(self, index):
        if self.istrain:
            idx = random.randint(0, self.file_num - 1)
        else:
            idx = index

        fid = self.file_list[idx]
        hsi_path = os.path.join(self.path, 'HSI', fid + '.mat')
        rgb_path = os.path.join(self.path, 'RGB', fid + '.mat')

        # load .mat using cache to avoid repeated disk I/O
        if fid in self._cache:
            hsi, rgb = self._cache[fid]
        else:
            data_h = sio.loadmat(hsi_path)
            # Harvard uses 'ref' typically; accept either
            if 'hsi' in data_h:
                hsi = data_h['hsi']
            elif 'ref' in data_h:
                hsi = data_h['ref']
            else:
                # fallback: find first 3D array
                hsi = None
                for v in data_h.values():
                    if isinstance(v, np.ndarray) and v.ndim == 3:
                        hsi = v
                        break
                if hsi is None:
                    raise RuntimeError(f"No 3D HSI array found in {hsi_path}")

            data_r = sio.loadmat(rgb_path)
            if 'rgb' in data_r:
                rgb = data_r['rgb']
            else:
                # try common alternatives
                for v in data_r.values():
                    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3:
                        rgb = v
                        break
                else:
                    raise RuntimeError(f"No 3-channel RGB found in {rgb_path}")
            # convert to float32 now and cache
            hsi = hsi.astype(np.float32)
            rgb = rgb.astype(np.float32)
            self._cache[fid] = (hsi, rgb)

        # Crop a random patch of size sizeI
        H, W, _ = hsi.shape
        if H < self.sizeI or W < self.sizeI:
            raise RuntimeError(f"Image {fid} smaller than patch size {self.sizeI}: {hsi.shape}")

        px = random.randint(0, H - self.sizeI)
        py = random.randint(0, W - self.sizeI)
        hr_hsi = hsi[px:px + self.sizeI, py:py + self.sizeI, :].copy()
        hr_msi = rgb[px:px + self.sizeI, py:py + self.sizeI, :].copy()

        # Data augmentation for training
        if self.istrain:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            for j in range(rotTimes):
                hr_hsi = np.rot90(hr_hsi)
                hr_msi = np.rot90(hr_msi)
            if vFlip:
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()
            if hFlip:
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        # compute LR using numpy FFT (fast on CPU workers) using precomputed fft_B when available
        if self._fft_B_np is None:
            # compute on the fly if not precomputed (rare)
            fft_B_np, _ = para_setting('gaussian_blur', self.factor, [self.sizeI, self.sizeI], sigma=2.0)
        else:
            fft_B_np = self._fft_B_np

        # hr_hsi: (H, W, C) -> compute per-channel FFT blur and downsample
        C = hr_hsi.shape[2]
        out_h = (self.sizeI - self._ds_offset) // self.factor
        out_w = (self.sizeI - self._ds_offset) // self.factor
        lr = np.zeros((C, out_h, out_w), dtype=np.float32)
        for c in range(C):
            f = np.fft.fft2(hr_hsi[:, :, c])
            M = f * fft_B_np
            Hz = np.fft.ifft2(M).real
            lr_c = Hz[self._ds_offset::self.factor, self._ds_offset::self.factor]
            lr[c, :lr_c.shape[0], :lr_c.shape[1]] = lr_c.astype(np.float32)

        # convert to torch tensors (CHW)
        hr_hsi_t = torch.from_numpy(hr_hsi.copy()).permute(2, 0, 1).float()
        hr_msi_t = torch.from_numpy(hr_msi.copy()).permute(2, 0, 1).float()
        lr_hsi = torch.from_numpy(lr).float()

        return lr_hsi, hr_msi_t, hr_hsi_t

    def __len__(self):
        return self.num
