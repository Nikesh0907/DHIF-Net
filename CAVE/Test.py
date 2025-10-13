import torch.utils.data as tud
import argparse
from Utils import *
from CAVE_Dataset import cave_dataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='/kaggle/input/cave-dataset-2/Data/Test/', type=str, help='path of the testing data (expects HSI/ and RGB/ under this folder)')
parser.add_argument("--sizeI", default=512, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=12, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
parser.add_argument("--seed", default=1, type=int, help='Random seed')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

key = 'Test.txt'
file_path = opt.data_path + key
# Fallback to repo list if Test.txt isn't present in Kaggle dataset
if not os.path.exists(file_path):
    file_path = os.path.join(os.path.dirname(__file__), 'Data/Test', key)
    print(f"Test list not found in data_path. Falling back to {file_path}")
file_list = loadpath(file_path, shuffle=False)
HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 12)

dataset = cave_dataset(opt, HR_HSI, HR_MSI, istrain=False)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)


# Try to load the latest checkpoint from Model directory; fallback to single file
ckpt_dir = "./Checkpoint/f8/Model"
ckpt_path = None
if os.path.isdir(ckpt_dir):
    last_epoch = findLastCheckpoint(save_dir=ckpt_dir)
    if last_epoch > 0:
        ckpt_path = os.path.join(ckpt_dir, f"model_{last_epoch:03d}.pth")
if ckpt_path is None:
    ckpt_path = "./Checkpoint/f8/model.pth"
print(f"Loading checkpoint: {ckpt_path}")
model = torch.load(ckpt_path)
model = model.eval()
model = model.cuda()


psnr_total = 0
k = 0
for j, (LR, RGB, HR) in enumerate(loader_train):
    with torch.no_grad():
        out = model(RGB.cuda(), LR.cuda())
        result = out
        result = result.clamp(min=0., max=1.)
    psnr = compare_psnr(result.cpu().detach().numpy(), HR.numpy(), data_range=1.0)
    psnr_total = psnr_total + psnr
    k = k + 1
    #
    # res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    # save_path = './Result/ssr/' + str(j + 1) + '.mat'
    # sio.savemat(save_path, {'res':res})

print(k)
print("Avg PSNR = %.4f" % (psnr_total/k))
