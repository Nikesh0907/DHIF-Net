from Model import HSI_Fusion
from CAVE_Dataset import cave_dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *
try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback: no-op tqdm
    def tqdm(x, **kwargs):
        return x


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


if __name__=="__main__":

    ## Model Config
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default='/kaggle/input/cave-dataset-2/Data/Train/', type=str,
                        help='Path of the training data (expects HSI/ and RGB/ under this folder)')
    parser.add_argument("--sizeI", default=96, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=4, type=int, help='Batch size')
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
    parser.add_argument("--seed", default=1, type=int, help='Random seed')
    parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    parser.add_argument("--epochs", default=500, type=int, help='Total number of epochs to run')
    parser.add_argument("--gpus", default=1, type=int, help='Number of GPUs to use with DataParallel (set >1 to use multiple GPUs if available)')
    parser.add_argument("--resume", dest="resume", action="store_true", help='Auto-resume from last checkpoints in default folder if available')
    parser.set_defaults(resume=True)
    parser.add_argument("--resume_path", type=str, default="", help='Path to a .pth checkpoint to resume from (stateful or weights). Overrides auto-resume if provided.')
    opt = parser.parse_args()

    # Configure visible GPUs BEFORE any cuda calls
    if opt.gpus is not None and opt.gpus > 0:
        # If more than one GPU requested, expose 0..gpus-1; otherwise do not restrict
        try:
            ids = ",".join(str(i) for i in range(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ids
        except Exception:
            pass

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print(opt)

    ## New model
    print("===> New Model")
    model = HSI_Fusion(Ch=31, stages=4, sf=opt.sf)

    ## set the number of parallel GPUs
    print("===> Setting GPU")
    model = dataparallel(model, opt.gpus)
    # Resolve device from model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    ## Load training data
    key = 'Train.txt'
    file_path = opt.data_path + key
    # Fallback: if Train.txt does not exist in the Kaggle input dataset, use the repo's default list
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), 'Data/Train', key)
        print(f"Train list not found in data_path. Falling back to {file_path}")
    file_list = loadpath(file_path)
    HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 20)

    ## Checkpoints and resume
    checkpoint_dir = "./Checkpoint/f8/Model"
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_ckpt = os.path.join(checkpoint_dir, "checkpoint_last.pth")
    best_ckpt = os.path.join(checkpoint_dir, "model_best.pth")

    ## Loss function
    criterion = nn.L1Loss()

    ## optimizer and scheduler
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8)

    # Determine resume starting point
    start_epoch = 0
    best_loss = float("inf")

    scheduler = None  # will be created/loaded below

    # 1) Highest priority: explicit resume_path
    if opt.resume_path and os.path.exists(opt.resume_path):
        print(f"Resuming from explicit path: {opt.resume_path}")
        loaded = torch.load(opt.resume_path, map_location=device)
        # Try stateful first (dict with model/optimizer/scheduler)
        if isinstance(loaded, dict) and ("model" in loaded or "state_dict" in loaded):
            state = loaded
            # Normalize to state_dict
            model_state = state.get("model") or state.get("state_dict") or state
            if isinstance(model_state, dict):
                try:
                    model.load_state_dict(model_state, strict=False)
                except Exception as e:
                    print(f"[Warn] Could not load model_state from resume_path: {e}")
            else:
                # In rare case "model" holds a full module
                try:
                    model = model_state
                except Exception:
                    pass
            # Optimizer/scheduler/epoch if present
            if "optimizer" in state:
                try:
                    optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
                except Exception as e:
                    print(f"[Warn] Could not load optimizer state from resume_path: {e}")
            if "epoch" in state:
                start_epoch = int(state["epoch"]) + 1
            if "best_loss" in state:
                best_loss = float(state["best_loss"]) or best_loss
            if "scheduler" in state:
                try:
                    scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95)
                    scheduler.load_state_dict(state["scheduler"])  # type: ignore[arg-type]
                except Exception as e:
                    print(f"[Warn] Could not load scheduler state from resume_path: {e}")
        elif isinstance(loaded, torch.nn.Module):
            model = loaded
        elif isinstance(loaded, dict):
            # Assume it's pure state_dict
            try:
                model.load_state_dict(loaded, strict=False)
            except Exception as e:
                print(f"[Warn] Could not load weights from resume_path: {e}")

    # 2) Next priority: auto stateful resume (checkpoint_last.pth)
    elif opt.resume and os.path.exists(last_ckpt):
        print(f"Resuming from stateful checkpoint: {last_ckpt}")
        state = torch.load(last_ckpt, map_location=device)
        # Load model weights
        if isinstance(state.get("model"), dict):
            model.load_state_dict(state["model"], strict=False)
        else:
            try:
                model = state["model"]
            except Exception:
                pass
        # Load optimizer/scheduler states if present
        if "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
            except Exception as e:
                print(f"[Warn] Could not load optimizer state: {e}")
        start_epoch = int(state.get("epoch", 0)) + 1
        best_loss = float(state.get("best_loss", best_loss))
        # Create scheduler then load its state
        scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95, last_epoch=start_epoch-1)
        if "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])  # type: ignore[arg-type]
            except Exception as e:
                print(f"[Warn] Could not load scheduler state: {e}")
    else:
        # Otherwise, try to resume from the latest per-epoch weights
        if opt.resume:
            initial_epoch = findLastCheckpoint(save_dir=checkpoint_dir)
            if initial_epoch > 0:
                latest_path = os.path.join(checkpoint_dir, f'model_{initial_epoch:03d}.pth')
                print('Resuming from latest weights: %s' % latest_path)
                loaded = torch.load(latest_path, map_location=device)
                if isinstance(loaded, torch.nn.Module):
                    model = loaded
                else:
                    try:
                        model.load_state_dict(loaded, strict=False)
                    except Exception as e:
                        print(f"[Warn] Could not load state_dict: {e}")
                start_epoch = initial_epoch
        # Initialize scheduler aligned to starting epoch if not created yet
        if scheduler is None:
            scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95, last_epoch=start_epoch-1)
    
    # Final guard: ensure scheduler exists for all code paths
    if 'scheduler' not in locals() or scheduler is None:
        scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95, last_epoch=start_epoch-1)

    ## pipline of training
    for epoch in range(start_epoch, opt.epochs):
        model.train()

        dataset = cave_dataset(opt, HR_HSI, HR_MSI)
        loader_train = tud.DataLoader(dataset, num_workers=1, batch_size=opt.batch_size, shuffle=True)

        epoch_loss = 0.0
        start_time = time.time()
        interrupted = False

        pbar = tqdm(enumerate(loader_train), total=len(loader_train), desc=f"Epoch {epoch+1}/{opt.epochs}", dynamic_ncols=True)
        try:
            for i, (LR, RGB, HR) in pbar:
                LR, RGB, HR = Variable(LR).to(device), Variable(RGB).to(device), Variable(HR).to(device)
                out = model(RGB, LR)

                loss = criterion(out, HR)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss = epoch_loss / (i + 1)
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{current_lr:.6e}")
        except KeyboardInterrupt:
            print("\n[Info] Interrupted by user. Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(checkpoint_dir, 'checkpoint_interrupt.pth'))
            interrupted = True

        # Step the scheduler at end-of-epoch (recommended order)
        scheduler.step()

        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / max(1, len(loader_train))
        print('epoch = %4d , avg_loss = %.10f , time = %4.2f s , lr = %.6e' % (
            epoch + 1, avg_loss, elapsed_time, optimizer.param_groups[0]['lr']))

        # Save stateful checkpoint for resume
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }, last_ckpt)

        # Save per-epoch weights-only snapshot and best
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_{epoch + 1:03d}.pth'))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_ckpt)

        if interrupted:
            print("[Info] Training stopped after saving interrupt checkpoint.")
            break
