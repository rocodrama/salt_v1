# src/train_hair_ldm.py
#
# Train unconditional hair ROI LDM (latent diffusion) on 256x256 crops.
# - Expects:
#     <train>/images/*.png
#     <val>/images/*.png
#
# Saves:
#   <out>/best/unet        (lowest val loss)
#   <out>/epoch_<N>/unet   (every --save_epoch)
#   <out>/final/unet
#
# NEW:
#   --sample N
#     When a checkpoint is saved (best / epoch_N / final), generate N sample images and save under:
#       <out>/<tag>/samples/sample_00000.png ...
#
# Example:
#   python src/train_hair_ldm.py --train dataset/train --val dataset/val --out hair_ldm_ckpt \
#       --epoch 50 --save_epoch 5 --batch 16 --lr 1e-4 --gpu 0 --sample 16 --sample_steps 50

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ImageFolder256(Dataset):
    def __init__(self, root_images_dir: str):
        self.root = Path(root_images_dir)
        self.paths = [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        self.paths.sort()

        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),                      # [0,1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return x


def set_device(gpu_arg: str) -> str:
    """
    gpu_arg examples:
      - "0", "1", ...
      - "cpu"
    """
    if gpu_arg.lower() == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        try:
            gpu_idx = int(gpu_arg)
            torch.cuda.set_device(gpu_idx)
            return f"cuda:{gpu_idx}"
        except Exception:
            return "cuda"

    return "cpu"


@torch.no_grad()
def encode_latents(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(x).latent_dist.sample()
    return latents * 0.18215  # SD convention


def save_ckpt(unet: UNet2DModel, out_dir: Path, tag: str):
    ckpt_dir = out_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(ckpt_dir / "unet"))
    with open(ckpt_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write("vae=stabilityai/sd-vae-ft-mse\n")
        f.write("latent_scale=0.18215\n")
        f.write("image_size=256\n")
        f.write("latent_size=32\n")
        f.write("latent_channels=4\n")
    print(f"[SAVE] {ckpt_dir}")
    return ckpt_dir


@torch.no_grad()
def sample_images(
    unet: UNet2DModel,
    vae: AutoencoderKL,
    device: str,
    out_dir: Path,
    n: int,
    steps: int,
    seed: int,
):
    """
    Generate unconditional samples from the current UNet.
    Saves to out_dir/sample_XXXXX.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sampling deterministic per checkpoint (and independent of training RNG)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps)

    # latent shape for 256x256 using SD VAE: (B,4,32,32)
    latents = torch.randn((n, 4, 32, 32), device=device, generator=g)

    unet.eval()
    vae.eval()

    for i, t in enumerate(scheduler.timesteps):
        noise_pred = unet(latents, t).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0 or (i + 1) == len(scheduler.timesteps):
            print(f"[SAMPLE] step {i+1}/{len(scheduler.timesteps)}")

    # decode
    latents = latents / 0.18215
    imgs = vae.decode(latents).sample  # [-1,1]
    imgs = (imgs.clamp(-1, 1) + 1) * 0.5  # [0,1]

    for i in range(n):
        save_image(imgs[i], str(out_dir / f"sample_{i:05d}.png"))

    print(f"[SAMPLE] saved {n} images -> {out_dir}")


def train_one_epoch(unet, vae, scheduler, opt, dl, device, epoch_idx, print_every=50):
    unet.train()
    running = 0.0
    n = 0

    for it, x in enumerate(dl, 1):
        x = x.to(device)

        with torch.no_grad():
            latents = encode_latents(vae, x)

        noise = torch.randn_like(latents)
        b = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (b,), device=device
        ).long()

        noisy = scheduler.add_noise(latents, noise, timesteps)
        pred = unet(noisy, timesteps).sample
        loss = F.mse_loss(pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += float(loss.item())
        n += 1

        if it % print_every == 0:
            print(f"[TRAIN] epoch={epoch_idx} iter={it}/{len(dl)} loss={loss.item():.6f}")

    return running / max(1, n)


@torch.no_grad()
def eval_one_epoch(unet, vae, scheduler, dl, device, epoch_idx, print_every=50):
    unet.eval()
    running = 0.0
    n = 0

    for it, x in enumerate(dl, 1):
        x = x.to(device)
        latents = encode_latents(vae, x)

        noise = torch.randn_like(latents)
        b = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (b,), device=device
        ).long()

        noisy = scheduler.add_noise(latents, noise, timesteps)
        pred = unet(noisy, timesteps).sample
        loss = F.mse_loss(pred, noise)

        running += float(loss.item())
        n += 1

        if it % print_every == 0:
            print(f"[VAL]   epoch={epoch_idx} iter={it}/{len(dl)} loss={loss.item():.6f}")

    return running / max(1, n)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", required=True, help="train split folder (contains images/)")
    ap.add_argument("--val", required=True, help="val split folder (contains images/)")
    ap.add_argument("--out", default="hair_ldm_ckpt", help="output folder")

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--save_epoch", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default="0", help='GPU index like "0", "1" ... or "cpu"')

    # NEW: sampling on checkpoint save
    ap.add_argument("--sample", type=int, default=0, help="generate N samples whenever a checkpoint is saved (0=off)")
    ap.add_argument("--sample_steps", type=int, default=50, help="DDPM sampling steps (default: 50)")
    ap.add_argument("--sample_seed", type=int, default=0, help="base seed for sampling (default: 0)")

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = set_device(args.gpu)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = Path(args.train) / "images"
    val_images_dir = Path(args.val) / "images"

    print(f"[INFO] device={device}")
    print(f"[INFO] train_images_dir={train_images_dir}")
    print(f"[INFO] val_images_dir={val_images_dir}")

    if not train_images_dir.exists():
        print(f"[ERROR] train images folder not found: {train_images_dir}")
        return
    if not val_images_dir.exists():
        print(f"[ERROR] val images folder not found: {val_images_dir}")
        return

    ds_train = ImageFolder256(str(train_images_dir))
    ds_val = ImageFolder256(str(val_images_dir))

    if len(ds_train) == 0:
        print("[ERROR] No training images found.")
        return
    if len(ds_val) == 0:
        print("[ERROR] No validation images found.")
        return

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"[INFO] train={len(ds_train)} val={len(ds_val)}")
    print(f"[INFO] batch={args.batch} lr={args.lr} epoch={args.epoch} save_epoch={args.save_epoch}")
    print(f"[INFO] sample={args.sample} sample_steps={args.sample_steps} sample_seed={args.sample_seed}")

    # Save run config
    with open(out_dir / "run_args.txt", "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")

    # VAE (frozen)
    # NOTE: needs internet the first time to download, unless cached.
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # UNet (trainable)
    unet = UNet2DModel(
        sample_size=32,   # 256 -> latent 32
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    best_val = 1e18

    def save_and_maybe_sample(tag: str, extra_seed_offset: int):
        ckpt_dir = save_ckpt(unet, out_dir, tag)
        if args.sample and args.sample > 0:
            sample_dir = ckpt_dir / "samples"
            # make seed differ across checkpoints but still deterministic
            seed = args.sample_seed + extra_seed_offset
            sample_images(
                unet=unet,
                vae=vae,
                device=device,
                out_dir=sample_dir,
                n=args.sample,
                steps=args.sample_steps,
                seed=seed,
            )

    for ep in range(1, args.epoch + 1):
        train_loss = train_one_epoch(unet, vae, scheduler, opt, dl_train, device, ep, print_every=50)
        val_loss = eval_one_epoch(unet, vae, scheduler, dl_val, device, ep, print_every=50)

        print(f"[EPOCH {ep}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            # offset: 100000 + epoch index (to separate from epoch ckpt seeds)
            save_and_maybe_sample("best", extra_seed_offset=100000 + ep)
            print(f"[BEST] val_loss improved -> {best_val:.6f}")

        # Periodic checkpoint
        if args.save_epoch > 0 and (ep % args.save_epoch == 0):
            save_and_maybe_sample(f"epoch_{ep}", extra_seed_offset=ep)

    # Final
    save_and_maybe_sample("final", extra_seed_offset=200000 + args.epoch)
    print("[DONE]")


if __name__ == "__main__":
    main()
