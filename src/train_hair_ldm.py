# train_hair_ldm.py
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ImageFolder256(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.paths = [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        self.paths.sort()

        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),               # [0,1]
            transforms.Normalize([0.5]*3, [0.5]*3),  # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cropped hair ROI folder (256x256 images)")
    ap.add_argument("--out", default="hair_ldm_ckpt", help="output folder")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    ds = ImageFolder256(args.data)
    if len(ds) == 0:
        print("[ERROR] No images found in data folder.")
        return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)

    print(f"[INFO] device={device}")
    print(f"[INFO] dataset size={len(ds)}")
    print(f"[INFO] batch={args.batch} lr={args.lr} steps={args.steps}")

    # 1) VAE: SD 계열의 사전학습 VAE를 가져와 latent로 압축/복원
    # - 인터넷 다운로드가 필요함(처음 1회). 오프라인이면 모델 파일을 미리 받아야 함.
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # 2) UNet: latent(4ch)에서 노이즈 예측
    # latent 크기는 보통 (B,4,32,32) (256 입력 기준)
    unet = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    step = 0
    dl_iter = iter(dl)

    while step < args.steps:
        try:
            x = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            x = next(dl_iter)

        x = x.to(device)

        with torch.no_grad():
            # encode -> latents
            latents = vae.encode(x).latent_dist.sample()
            latents = latents * 0.18215  # SD 관례 스케일

        # sample noise + timesteps
        noise = torch.randn_like(latents)
        b = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        pred = unet(noisy_latents, timesteps).sample
        loss = F.mse_loss(pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1

        if step % 50 == 0:
            print(f"[STEP {step}] loss={loss.item():.6f}")

        if step % args.save_every == 0:
            ckpt_dir = Path(args.out) / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(str(ckpt_dir / "unet"))
            # VAE는 고정이지만, 샘플링 스크립트 편하게 하려면 이름 기록
            with open(ckpt_dir / "meta.txt", "w", encoding="utf-8") as f:
                f.write("vae=stabilityai/sd-vae-ft-mse\n")
                f.write("latent_scale=0.18215\n")
            print(f"[SAVE] {ckpt_dir}")

    # final save
    final_dir = Path(args.out) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(final_dir / "unet"))
    with open(final_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write("vae=stabilityai/sd-vae-ft-mse\n")
        f.write("latent_scale=0.18215\n")
    print(f"[DONE] saved to {final_dir}")


if __name__ == "__main__":
    main()
