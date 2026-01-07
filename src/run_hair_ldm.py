# sample_hair_ldm.py
import argparse
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="e.g. hair_ldm_ckpt/final/unet")
    ap.add_argument("--out", default="samples", help="output folder")
    ap.add_argument("--n", type=int, default=16, help="number of samples")
    ap.add_argument("--steps", type=int, default=50, help="ddpm sampling steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    unet = UNet2DModel.from_pretrained(args.ckpt).to(device)
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.steps)

    # latent shape: (B,4,32,32)
    latents = torch.randn((args.n, 4, 32, 32), device=device)

    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = unet(latents, t).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        if (i + 1) % 10 == 0:
            print(f"[SAMPLE] step {i+1}/{len(scheduler.timesteps)}")

    with torch.no_grad():
        latents = latents / 0.18215
        imgs = vae.decode(latents).sample  # [-1,1]
        imgs = (imgs.clamp(-1, 1) + 1) * 0.5  # [0,1]

    # save with torchvision to avoid extra deps
    from torchvision.utils import save_image
    for i in range(args.n):
        save_image(imgs[i], str(Path(args.out) / f"sample_{i:04d}.png"))
    print(f"[DONE] saved {args.n} images to {args.out}")


if __name__ == "__main__":
    main()
