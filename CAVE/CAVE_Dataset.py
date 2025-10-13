import torch.utils.data as tud
from Utils import *


class cave_dataset(tud.Dataset):
    def __init__(self, opt, HR_HSI, HR_MSI, istrain = True):
        super(cave_dataset, self).__init__()
        self.path = opt.data_path
        self.istrain  =  istrain
        self.factor = opt.sf
        if istrain:
            self.num = opt.trainset_num
            self.file_num = 20
            self.sizeI = opt.sizeI
        else:
            self.num = opt.testset_num
            self.file_num = 12
            self.sizeI = 512
        self.HR_HSI, self.HR_MSI = HR_HSI, HR_MSI

    def H_z(self, z, factor, fft_B):
        # Use torch.fft for compatibility with newer PyTorch
        device = z.device
        if len(z.shape) == 3:
            ch, h, w = z.shape
            f = torch.fft.fft2(z, dim=(-2, -1))
            fft_B_c = torch.complex(fft_B[:, :, 0].to(device), fft_B[:, :, 1].to(device))  # [h,w]
            M = f * fft_B_c  # broadcast over channels
            Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
            x = Hz[:, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            f = torch.fft.fft2(z, dim=(-2, -1))
            fft_B_c = torch.complex(fft_B[:, :, 0].to(device), fft_B[:, :, 1].to(device)).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
            M = f * fft_B_c
            Hz = torch.fft.ifft2(M, dim=(-2, -1)).real
            x = Hz[:, :, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        return x

    def __getitem__(self, index):
        if self.istrain == True:
            index1   = random.randint(0, self.file_num-1)
        else:
            index1 = index

        sigma = 2.0
        HR_HSI = self.HR_HSI[:,:,:,index1]
        HR_MSI = self.HR_MSI[:,:,:,index1]

        sz = [self.sizeI, self.sizeI]
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        px      = random.randint(0, 512-self.sizeI)
        py      = random.randint(0, 512-self.sizeI)
        hr_hsi  = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hr_msi  = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

        if self.istrain == True:
            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hr_hsi  =  np.rot90(hr_hsi)
                hr_msi  =  np.rot90(hr_msi)

            # Random vertical Flip
            for j in range(vFlip):
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2,0,1).unsqueeze(0)
        hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2,0,1).unsqueeze(0)
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
        lr_hsi = torch.FloatTensor(lr_hsi)

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)

        return lr_hsi, hr_msi, hr_hsi

    def __len__(self):
        return self.num
