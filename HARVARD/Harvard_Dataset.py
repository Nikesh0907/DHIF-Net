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
            # for testing, by default we use a 512 crop (historical CAVE default).
            # To run full-image evaluation on Harvard, pass --sizeI 0 to the test script
            # and the dataset will return full-resolution images for testing.
            self.sizeI = opt.sizeI if hasattr(opt, 'sizeI') else 512

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

        # If sizeI==0 and this is testing, return the full image (no crop)
        H, W, _ = hsi.shape
        if not self.istrain and self.sizeI == 0:
            hr_hsi = hsi.copy()
            hr_msi = rgb.copy()
            cur_size_h, cur_size_w = H, W
        else:
            # Crop a random patch of size sizeI
            if H < self.sizeI or W < self.sizeI:
                raise RuntimeError(f"Image {fid} smaller than patch size {self.sizeI}: {hsi.shape}")

            px = random.randint(0, H - self.sizeI)
            py = random.randint(0, W - self.sizeI)
            hr_hsi = hsi[px:px + self.sizeI, py:py + self.sizeI, :].copy()
            hr_msi = rgb[px:px + self.sizeI, py:py + self.sizeI, :].copy()
            cur_size_h, cur_size_w = hr_hsi.shape[0], hr_hsi.shape[1]

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

        # convert to torch and compute LR via para_setting + H_z
        # prepare fft_B; use actual current size (supports full-image test when sizeI==0)
        sz = [cur_size_h, cur_size_w]
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma=2.0)
        # fft_B is numpy complex; convert to torch complex representation in Dataset
        fft_B_t = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)), 2)

        hr_hsi_t = torch.FloatTensor(hr_hsi.copy()).permute(2, 0, 1).unsqueeze(0)
        hr_msi_t = torch.FloatTensor(hr_msi.copy()).permute(2, 0, 1).unsqueeze(0)

        lr_hsi = self.H_z(hr_hsi_t, self.factor, fft_B_t)
        lr_hsi = torch.FloatTensor(lr_hsi).squeeze(0)
        hr_hsi_t = hr_hsi_t.squeeze(0)
        hr_msi_t = hr_msi_t.squeeze(0)

        return lr_hsi, hr_msi_t, hr_hsi_t

    def __len__(self):
        return self.num
