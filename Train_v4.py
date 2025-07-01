import os
import time
import random
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import vgg16_bn
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import piq

from model.HiIR import HiIR

# -----------------------------
# CONFIGURACIÓN DE LOGGING
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

info_handler = logging.FileHandler('manga_restoration_info.log')
info_handler.setLevel(logging.INFO)
error_handler = logging.FileHandler('manga_restoration_error.log')
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

# -----------------------------
# 1. DATASET: MangaRestorationDataset
# -----------------------------
class MangaRestorationDataset(Dataset):
    """
    Dataset para manga en escala de grises (1 canal):
      - Cada imagen como un solo canal L.
    """
    def __init__(self, manga_root_dir: str, patch_size: int = 64, augmentation_factor: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.aug_factor = augmentation_factor

        logger.info(f"Cargando imágenes desde: {manga_root_dir}")
        self.image_paths = sorted(Path(manga_root_dir).rglob('*.png')) + \
                           sorted(Path(manga_root_dir).rglob('*.jpg'))
        if not self.image_paths:
            logger.error(f"No se encontraron imágenes en {manga_root_dir}")
            raise RuntimeError(f"[ERROR Dataset] No hay imágenes en {manga_root_dir}")
        logger.info(f"Imágenes encontradas: {len(self.image_paths)} archivos")

    def __len__(self):
        return len(self.image_paths) * self.aug_factor

    def __getitem__(self, idx):
        base_idx = idx // self.aug_factor
        img_path = self.image_paths[base_idx]

        # Carga y conversión a grayscale
        img = Image.open(img_path).convert('L')
        arr = np.array(img)

        # Asegurar tamaño mínimo
        if img.width < self.patch_size or img.height < self.patch_size:
            img = img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            arr = np.array(img)

        # Recorte aleatorio
        x = random.randint(0, arr.shape[1] - self.patch_size)
        y = random.randint(0, arr.shape[0] - self.patch_size)
        patch = arr[y:y + self.patch_size, x:x + self.patch_size]

        # Generar HR y LR
        img_hr = Image.fromarray(patch)
        lr_size = (self.patch_size // 4, self.patch_size // 4)
        img_lr = img_hr.resize(lr_size, Image.BICUBIC)

        hr_np = np.array(img_hr).astype(np.float32) / 255.0
        lr_np = np.array(img_lr).astype(np.float32) / 255.0
        hr_t = torch.from_numpy(hr_np).unsqueeze(0)
        lr_t = torch.from_numpy(lr_np).unsqueeze(0)

        return {'degraded': lr_t, 'clean': hr_t, 'manga_name': img_path.stem}

# -----------------------------
# 2. MÉTRICAS: PSNR y SSIM
# -----------------------------
def compute_metrics_tensor(sr: torch.Tensor, hr: torch.Tensor):
    try:
        psnr = piq.psnr(sr, hr, data_range=1.0, reduction='mean').item()
        ssim = piq.ssim(sr, hr, data_range=1.0, reduction='mean').item()
        return psnr, ssim
    except Exception as e:
        logger.error(f"Error al computar métricas: {e}")
        raise

# -----------------------------
# 3. PÉRDIDAS ADICIONALES
# -----------------------------
def sobel_gradients(x):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=x.dtype,device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=x.dtype,device=x.device).view(1,1,3,3)
    Gx = F.conv2d(x, sobel_x.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
    Gy = F.conv2d(x, sobel_y.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
    return Gx, Gy

def gradient_loss(sr, hr):
    Gx_sr, Gy_sr = sobel_gradients(sr)
    Gx_hr, Gy_hr = sobel_gradients(hr)
    return F.l1_loss(Gx_sr, Gx_hr) + F.l1_loss(Gy_sr, Gy_hr)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg16_bn(pretrained=True).features[:16].eval().to(device)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)

    def forward(self, sr, hr):
        sr_n = (sr - self.mean)/self.std
        hr_n = (hr - self.mean)/self.std
        return F.mse_loss(self.vgg(sr_n), self.vgg(hr_n))

# -----------------------------
# 4. TRAINER
# -----------------------------
class MangaTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.total_steps = 0
        self.warmup_steps = 50000
        self.decay_step = 200000
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, betas=(0.9,0.999), weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.perc = PerceptualLoss(self.device)
        logger.info("MangaTrainer inicializado.")

    def _lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        num_decays = (step - self.warmup_steps) // self.decay_step
        return 0.5 ** num_decays

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = running_psnr = running_ssim = cnt = 0
        for batch in dataloader:
            lr = batch['degraded'].to(self.device)
            hr = batch['clean'].to(self.device)
            sr = self.model(lr)
            pixel_l = self.l1(sr, hr)
            mse_l = self.mse(sr, hr)
            ssim_l = 1.0 - piq.ssim(sr, hr, data_range=1.0, reduction='mean')
            grad_l = gradient_loss(sr, hr)
            perc_l = self.perc(sr, hr)
            loss = pixel_l + 0.1*mse_l + 0.1*ssim_l + 0.2*grad_l + 0.01*perc_l
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.total_steps += 1
            running_loss += loss.item()
            with torch.no_grad(): psnr_b, ssim_b = compute_metrics_tensor(sr, hr)
            running_psnr += psnr_b
            running_ssim += ssim_b
            cnt += 1
        avg_loss = running_loss / cnt
        avg_psnr = running_psnr / cnt
        avg_ssim = running_ssim / cnt
        logger.info(f"Train Epoch: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        return avg_loss, avg_psnr, avg_ssim

    def validate(self, dataloader, max_batches=5):
        self.model.eval()
        val_loss = val_psnr = val_ssim = cnt = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches: break
                lr = batch['degraded'].to(self.device)
                hr = batch['clean'].to(self.device)
                sr = self.model(lr)
                pixel_l = self.l1(sr, hr)
                mse_l = self.mse(sr, hr)
                ssim_l = 1.0 - piq.ssim(sr, hr, data_range=1.0, reduction='mean')
                grad_l = gradient_loss(sr, hr)
                perc_l = self.perc(sr, hr)
                loss = pixel_l + 0.1*mse_l + 0.1*ssim_l + 0.2*grad_l + 0.01*perc_l
                val_loss += loss.item()
                psnr_b, ssim_b = compute_metrics_tensor(sr, hr)
                val_psnr += psnr_b
                val_ssim += ssim_b
                cnt += 1
        if cnt == 0:
            return None, None, None
        avg_loss = val_loss / cnt
        avg_psnr = val_psnr / cnt
        avg_ssim = val_ssim / cnt
        logger.info(f"Validation: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        return avg_loss, avg_psnr, avg_ssim

    def train(self, num_epochs, batch_size, num_workers, save_path):
        try:
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)
            val_loader = None
            if self.val_dataset is not None:
                val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)
            logger.info(f"Inicio entrenamiento: epochs={num_epochs}, batch_size={batch_size}")
            for epoch in range(1, num_epochs+1):
                start_time = time.time()
                tl, tp, ts = self.train_epoch(train_loader)
                vl = vp = vs = None
                if val_loader is not None:
                    vl, vp, vs = self.validate(val_loader)
                elapsed = time.time() - start_time
                logger.info(f"Epoch {epoch}/{num_epochs} completada en {elapsed:.1f}s")
                if epoch % 10 == 0:
                    ckpt_path = f"{save_path}_ep{epoch}.pth"
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info(f"Checkpoint guardado en: {ckpt_path}")
            final_path = f"{save_path}_final.pth"
            torch.save(self.model.state_dict(), final_path)
            logger.info(f"Modelo final guardado en: {final_path}")
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            raise

# -----------------------------
# 5. MAIN
# -----------------------------
def main():
    manga_root = "./dataset"
    patch_size = 64
    aug_factor = 1
    train_split = 0.8
    batch_size = 256
    num_workers = 0
    num_epochs = 120
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    logger.info(f"Dispositivo: {device}")
    try:
        full_ds = MangaRestorationDataset(manga_root, patch_size, aug_factor)
        total = len(full_ds)
        train_len = int(total * train_split)
        val_len = total - train_len
        train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                         generator=torch.Generator().manual_seed(42))
        logger.info(f"Dataset cargado: total={total}, train={train_len}, val={val_len}")

        model_args = dict(
            img_size=patch_size,
            in_chans=1,
            out_chans=1,
            embed_dim=96,
            depths=16,
            num_heads=12,
            patch_size=8,
            window_size=2,
            mlp_ratio=8.0,
            qkv_bias=True,
            dropout=0.0
        )
        model = HiIR(**model_args).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Modelo HiIR instanciado con parámetros entrenables: {total_params:,}")

        trainer = MangaTrainer(model, train_ds, val_ds, device)
        trainer.train(num_epochs, batch_size, num_workers, "./models_v4/manga_hiir")
    except Exception as e:
        logger.error(f"Error en main: {e}")
        raise

if __name__ == "__main__":
    main()