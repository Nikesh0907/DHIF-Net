import numpy as np
import scipy.io as sio
import os
import glob
import torch
import torch.nn as nn
import skimage.measure as measure
import torch.nn.functional as F
import cv2
import Pypher
import random
import re
import math
try:
    # Prefer modern skimage API
    from skimage.metrics import structural_similarity as _ssim_metric
except Exception:
    _ssim_metric = None


def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)

def psnr(img1, img2):
   mse = np.mean((img1/255. - img2/255.) ** 2)
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def PSNR_GPU(im_true, im_fake):
    im_true *= 255
    im_fake *= 255
    im_true = im_true.round()
    im_fake = im_fake.round()
    data_range = 255
    esp = 1e-12
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone()
    Ifake = im_fake.clone()
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum() / (C*H*W)
    psnr = 10. * np.log((data_range**2)/(err.data + esp)) / np.log(10.)
    return psnr

def PSNR_Nssr(im_true, im_fake):
    mse = ((im_true - im_fake)**2).mean()
    psnr = 10. * np.log10(1/mse)
    return psnr

def c_psnr(im1, im2):
    '''
    Compute PSNR
    :param im1: input image 1 ndarray ranging [0,1]
    :param im2: input image 2 ndarray ranging [0,1]
    :return: psnr=-10*log(mse(im1,im2))
    '''
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_psnr(im1, im2)

def c_ssim(im1, im2):
    '''
    Compute PSNR
    :param im1: input image 1 ndarray ranging [0,1]
    :param im2: input image 2 ndarray ranging [0,1]
    :return: psnr=-10*log(mse(im1,im2))
    '''
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)

    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_ssim(im1, im2, win_size=11, data_range=1, gaussian_weights=True)


# ---------- New metrics: SAM, ERGAS, SSIM (numpy-based) ----------
def _ensure_chw_numpy(arr):
    """Ensure numpy array shape is (C, H, W). Accepts (B,C,H,W), (C,H,W), or (H,W,C).
    Uses conservative heuristics to distinguish CHW vs HWC.
    """
    a = np.asarray(arr)
    if a.ndim == 4:
        # Prefer NCHW if channel dim small (<=64); else if NHWC with last dim small
        if a.shape[1] <= 64:
            a = a[0]  # NCHW -> CHW
        elif a.shape[-1] <= 64:
            a = np.transpose(a[0], (2, 0, 1))  # NHWC -> CHW
        else:
            a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D or 4D array, got shape {a.shape}")
    # At this point a is 3D. Decide if CHW or HWC.
    C, H, W = a.shape[0], a.shape[1], a.shape[2]
    if C <= 64 and H >= 8 and W >= 8:
        # Likely CHW already
        return a
    if a.shape[-1] <= 64 and a.shape[0] >= 8 and a.shape[1] >= 8:
        # HWC -> CHW
        return np.transpose(a, (2, 0, 1))
    # Fallback: assume CHW
    return a


def compute_sam(im_true, im_test, eps=1e-12, as_degrees=True):
    """
    Spectral Angle Mapper (per-pixel angle between spectra), averaged over all pixels.
    Inputs can be (B,C,H,W), (C,H,W), or (H,W,C) with values typically in [0,1].
    Returns the mean angle (degrees by default).
    """
    x = _ensure_chw_numpy(im_true).astype(np.float64)
    y = _ensure_chw_numpy(im_test).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in SAM: {x.shape} vs {y.shape}")
    C, H, W = x.shape
    x_flat = x.reshape(C, -1)
    y_flat = y.reshape(C, -1)
    nom = np.sum(x_flat * y_flat, axis=0)
    denom = np.linalg.norm(x_flat, axis=0) * np.linalg.norm(y_flat, axis=0)
    cosang = np.clip(nom / np.maximum(denom, eps), -1.0, 1.0)
    ang = np.arccos(cosang)
    # Mask invalid (zero vectors) to avoid skewing average
    valid = np.isfinite(ang)
    if as_degrees:
        ang = ang / np.pi * 180.0
    return float(np.mean(ang[valid]))


def compute_ergas(im_true, im_test, scale=8, eps=1e-12):
    """
    ERGAS metric (lower is better).
    ERGAS = 100 * (h/l) * sqrt( (1/N) * sum_i (RMSE_i / mu_i)^2 )
    where h/l is the scale ratio (use sf for upsampling factor), N is band count.
    """
    x = _ensure_chw_numpy(im_true).astype(np.float64)
    y = _ensure_chw_numpy(im_test).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in ERGAS: {x.shape} vs {y.shape}")
    C = x.shape[0]
    # Per-band RMSE and mean of reference
    rmse = np.sqrt(np.mean((x - y) ** 2, axis=(1, 2)))
    mu = np.mean(x, axis=(1, 2))
    term = (rmse / np.maximum(mu, eps)) ** 2
    # ERGAS = 100 * (h/l) * sqrt(mean((RMSE_i / mu_i)^2)), where h/l = 1/scale
    ergas = 100.0 * (1.0 / float(scale)) * np.sqrt(np.mean(term))
    return float(ergas)


def compute_ssim(im_true, im_test, data_range=1.0, gaussian_weights=True, win_size=11):
    """
    Mean SSIM across bands. If arrays are 2D, computes single SSIM. For hyperspectral,
    computes SSIM per band and returns the average.
    """
    x = np.asarray(im_true)
    y = np.asarray(im_test)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in SSIM: {x.shape} vs {y.shape}")
    # 2D case
    if x.ndim == 2:
        if _ssim_metric is not None:
            return float(_ssim_metric(x, y, data_range=data_range, gaussian_weights=gaussian_weights, win_size=win_size))
        else:
            # Fallback to deprecated API
            return float(measure.compare_ssim(x, y, data_range=data_range, gaussian_weights=gaussian_weights, win_size=win_size))
    # 3D or 4D
    x_chw = _ensure_chw_numpy(x)
    y_chw = _ensure_chw_numpy(y)
    C = x_chw.shape[0]
    scores = []
    for c in range(C):
        Xa = x_chw[c]
        Ya = y_chw[c]
        if _ssim_metric is not None:
            s = _ssim_metric(Xa, Ya, data_range=data_range, gaussian_weights=gaussian_weights, win_size=win_size)
        else:
            s = measure.compare_ssim(Xa, Ya, data_range=data_range, gaussian_weights=gaussian_weights, win_size=win_size)
        scores.append(float(s))
    return float(np.mean(scores))

def batch_PSNR(im_true, im_fake, data_range):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clone().resize_(N, C*H*W)
    Ifake = im_fake.clone().resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)

def PSNR_GPU(im_true, im_fake):
    im_true *= 255
    im_fake *= 255
    im_true = im_true.round()
    im_fake = im_fake.round()
    data_range = 255
    esp = 1e-12
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone()
    Ifake = im_fake.clone()
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum() / (C*H*W)
    psnr = 10. * np.log((data_range**2)/(err.data + esp)) / np.log(10.)
    return psnr

def batch_SAM_GPU(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clone().resize_(N, C, H*W)
    Ifake = im_fake.clone().resize_(N, C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=1).resize_(N, H*W)
    denom1 = torch.pow(Itrue,2).sum(dim=1).sqrt_().resize_(N, H*W)
    denom2 = torch.pow(Ifake,2).sum(dim=1).sqrt_().resize_(N, H*W)
    sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(N, H*W)
    sam = sam / np.pi * 180
    sam = torch.sum(sam) / (N*H*W)
    return sam

def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    esp = 1e-12
    Itrue = im_true.clone()#.resize_(C, H*W)
    Ifake = im_fake.clone()#.resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum

def batch_SAM_CPU(im_true, im_fake):
    I_true = im_true.data.cpu().numpy()
    I_fake = im_fake.data.cpu().numpy()
    N = I_true.shape[0]
    C = I_true.shape[1]
    H = I_true.shape[2]
    W = I_true.shape[3]
    batch_sam = 0
    for i in range(N):
        true = I_true[i,:,:,:].reshape(C, H*W)
        fake = I_fake[i,:,:,:].reshape(C, H*W)
        nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
        denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
        denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
        sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
        sam = sam/np.pi*180
        # ignore pixels that have zero norm
        idx = (np.isfinite(sam))
        batch_sam += np.sum(sam[idx])/np.sum(idx)
        if np.sum(~idx) != 0:
            print("waring: some values were ignored when computing SAM")
    return batch_sam/N

def SAM_CPU(im_true, im_fake):
    I_true = im_true.data.cpu().numpy()
    I_fake = im_fake.data.cpu().numpy()
    N = I_true.shape[0]
    C = I_true.shape[1]
    H = I_true.shape[2]
    W = I_true.shape[3]
    batch_sam = 0
    for i in range(N):
        true = I_true[i,:,:,:].reshape(C, H*W)
        fake = I_fake[i,:,:,:].reshape(C, H*W)
        nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
        denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
        denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
        sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
        sam = sam/np.pi*180
        # ignore pixels that have zero norm
        idx = (np.isfinite(sam))
        batch_sam += np.sum(sam[idx])/np.sum(idx)
        if np.sum(~idx) != 0:
            print("waring: some values were ignored when computing SAM")
    return batch_sam/N


# Convert an image into patches
def Im2Patch(img,img2, win, stride=1, istrain=False): # Based on code written by Shuhang Gu (cssgu@comp.polyu.edu.hk)
    k = 0
    endw = img.shape[0]
    endh = img.shape[1]
    if endw<win or endh<win:
        return None,None
    patch = img[0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[0] * patch.shape[1]
    # TotalPatNum = (img.shape[0]-win+1)*(img.shape[1]-win+1)
    Y = np.zeros([win*win,TotalPatNum])
    Y2 = np.zeros([win*win,TotalPatNum])
    for i in range(win):
        for j in range(win):
            patch = img[i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[k,:] = np.array(patch[:]).reshape(TotalPatNum)
            patch2 = img2[i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y2[k, :] = np.array(patch2[:]).reshape(TotalPatNum)
            k = k + 1
    if istrain:
        return Y.reshape([win, win, TotalPatNum]), Y2.reshape([win, win, TotalPatNum])
    else:
        return Y.transpose().reshape([TotalPatNum,win,win,1]), Y2.transpose().reshape([TotalPatNum,win,win,1])

def para_setting(kernel_type,sf,sz,sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = Pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


def H_z(z, factor, fft_B ):
    #     z  [31 , 96 , 96]
    #     ch, h, w = z.shape
    # Use torch.fft APIs (rfft/irfft deprecated)
    device = z.device
    if len(z.shape) == 3:
        ch, h, w = z.shape
        f = torch.fft.fft2(z, dim=(-2, -1))
        fft_B_c = torch.complex(fft_B[:, :, 0].to(device), fft_B[:, :, 1].to(device))
        M = f * fft_B_c
        Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
        x = Hz[:, int(factor // 2)::factor, int(factor // 2)::factor]
    elif len(z.shape) == 4:
        bs, ch, h, w = z.shape
        f = torch.fft.fft2(z, dim=(-2, -1))
        fft_B_c = torch.complex(fft_B[:, :, 0].to(device), fft_B[:, :, 1].to(device)).unsqueeze(0).unsqueeze(0)
        M = f * fft_B_c
        Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
        x = Hz[:, :, int(factor // 2)::factor, int(factor // 2)::factor]
    return x

def HT_y(y, sf, fft_BT):
    device = y.device
    if len(y.shape) == 3:
        ch, w, h = y.shape
        z = F.pad(y, [0, 0, 0, 0, 0, sf * sf - 1], "constant", value=0)
        z = F.pixel_shuffle(z, upscale_factor=sf).view(1, ch, w * sf, h * sf)

        f = torch.fft.fft2(z, dim=(-2, -1))
        fft_BT_c = torch.complex(fft_BT[:, :, 0].to(device), fft_BT[:, :, 1].to(device))
        # broadcast fft_BT over ch as needed
        fft_BT_c = fft_BT_c.unsqueeze(0).repeat(ch, 1, 1)
        # Match f shape [1,ch,h,w] by unsqueeze
        M = f * fft_BT_c.unsqueeze(0)
        Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
    elif len(y.shape) == 4:
        bs, ch, w, h = y.shape
        z = y.view(-1, 1, w, h)
        z = F.pad(z, [0, 0, 0, 0, 0, sf * sf - 1, 0, 0], "constant", value=0)
        z = F.pixel_shuffle(z, upscale_factor=sf).view(bs, ch, w * sf, h * sf)

        f = torch.fft.fft2(z, dim=(-2, -1))
        fft_BT_c = torch.complex(fft_BT[:, :, 0].to(device), fft_BT[:, :, 1].to(device)).unsqueeze(0).unsqueeze(0)
        M = f * fft_BT_c
        Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
    return Hz

def dataparallel(model, ngpus, gpu0=0):
    """
    Wrap    model with DataParallel if CUDA is available and requested; otherwise run on CPU.
    This avoids hard-failing in environments without GPU (e.g., Kaggle CPU sessions).
    """
    has_cuda = torch.cuda.is_available()
    if has_cuda and ngpus >= 1 and torch.cuda.device_count() >= gpu0 + ngpus:
        gpu_list = list(range(gpu0, gpu0 + ngpus))
        if ngpus > 1:
            if not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
            else:
                model = model.cuda()
        else:
            model = model.cuda()
        return model
    else:
        print("[Warning] CUDA not available or insufficient GPUs; running on CPU.")
        return model.cpu()

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    epochs = []
    for file_ in file_list:
        # Only match numeric suffixes like model_001.pth
        m = re.search(r"model_(\d+)\.pth$", os.path.basename(file_))
        if m:
            try:
                epochs.append(int(m.group(1)))
            except ValueError:
                pass
    return max(epochs) if epochs else 0

def prepare_data(path, file_list, file_num):
    HR_HSI = np.zeros((((512,512,31,file_num))))
    HR_MSI = np.zeros((((512,512,3,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path, 'HSI/') + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['hsi']

        ####  get HrMSI
        path2 = os.path.join(path, 'RGB/') + HR_code + '.mat'
        data = sio.loadmat(path2)
        HR_MSI[:,:,:,idx] = data['rgb']
    return HR_HSI, HR_MSI

def loadpath(pathlistfile,shuffle=True):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    if shuffle==True:
        random.shuffle(pathlist)
    return pathlist


