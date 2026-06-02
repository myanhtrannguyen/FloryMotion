from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.image_style_tranform.dataset import CartoonGANEvalDataset, CartoonGANTrainDataset
from source.image_style_tranform.models.cartoon_gan.model import build_cartoon_gan
from source.image_style_tranform.models.cartoon_gan.utils import (
    save_image_grid,
    save_json,
    style_edge_fake,
    total_variation_loss,
)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cartoon_gan_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_history_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[-1].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def discriminator_loss(
    discriminator: nn.Module,
    real_style: torch.Tensor,
    generated: torch.Tensor,
    edge_fake: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    real_pred = discriminator(real_style)
    fake_pred = discriminator(generated.detach())
    edge_pred = discriminator(edge_fake.detach())

    real_loss = criterion(real_pred, torch.ones_like(real_pred))
    fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
    edge_loss = criterion(edge_pred, torch.zeros_like(edge_pred))
    return real_loss + 0.5 * (fake_loss + edge_loss)


def generator_loss(
    discriminator: nn.Module,
    photo: torch.Tensor,
    generated: torch.Tensor,
    adversarial_criterion: nn.Module,
    content_weight: float,
    tv_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    pred = discriminator(generated)
    adv_loss = adversarial_criterion(pred, torch.ones_like(pred))
    content_loss = torch.mean(torch.abs(generated - photo))
    tv_loss = total_variation_loss(generated)
    loss = adv_loss + content_weight * content_loss + tv_weight * tv_loss
    parts = {
        "g_adv": float(adv_loss.detach().cpu().item()),
        "content": float(content_loss.detach().cpu().item()),
        "tv": float(tv_loss.detach().cpu().item()),
    }
    return loss, parts


@torch.no_grad()
def save_validation_samples(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_images: int = 8,
) -> None:
    generator.eval()
    samples = []
    for photos, _ in loader:
        photos = photos.to(device)
        generated = generator(photos)
        for photo, cartoon in zip(photos, generated):
            samples.extend([photo.cpu(), cartoon.cpu()])
            if len(samples) >= max_images * 2:
                save_image_grid(samples, output_dir / f"epoch_{epoch:03d}.jpg", columns=2)
                generator.train()
                return
    save_image_grid(samples, output_dir / f"epoch_{epoch:03d}.jpg", columns=2)
    generator.train()


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    args: argparse.Namespace,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "args": vars(args),
            "history": history,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CartoonGAN on animeGAN Hayao or Shinkai style images.")
    parser.add_argument("--data-root", type=Path, default=Path("data/animeGAN"))
    parser.add_argument("--domain", choices=["Hayao", "Shinkai"], default="Hayao")
    parser.add_argument(
        "--photo-dir",
        type=Path,
        default=Path("val"),
        help="Photo/content folder. Relative paths are resolved from --data-root.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("source/image_style_tranform/models/cartoon_gan"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--content-weight", type=float, default=10.0)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--generator-blocks", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50, help="Log training losses every N steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = args.output_root / args.domain
    sample_dir = run_dir / "samples"
    checkpoint_dir = run_dir / "checkpoints"
    logger = setup_logger(run_dir / "train.log")

    train_dataset = CartoonGANTrainDataset(
        data_root=args.data_root,
        domain=args.domain,
        photo_dir=args.photo_dir,
        image_size=args.image_size,
    )
    val_dataset = CartoonGANEvalDataset(
        data_root=args.data_root,
        mode="val",
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info("Starting CartoonGAN training")
    logger.info("domain=%s data_root=%s photo_dir=%s", args.domain, args.data_root, args.photo_dir)
    logger.info(
        "train_size=%d val_size=%d style_dir=%s",
        len(train_dataset),
        len(val_dataset),
        train_dataset.style_dir,
    )
    logger.info(
        "epochs=%d batch_size=%d image_size=%d lr=%.6f device=%s",
        args.epochs,
        args.batch_size,
        args.image_size,
        args.lr,
        device,
    )

    generator, discriminator = build_cartoon_gan(generator_blocks=args.generator_blocks)
    generator.to(device)
    discriminator.to(device)

    adversarial_criterion = nn.MSELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        generator.train()
        discriminator.train()
        running = {"d_loss": 0.0, "g_loss": 0.0, "g_adv": 0.0, "content": 0.0, "tv": 0.0}

        for step, (photo, style, _) in enumerate(train_loader, start=1):
            step_start = time.time()
            photo = photo.to(device)
            style = style.to(device)

            generated = generator(photo)
            edge_fake = style_edge_fake(style)

            optimizer_d.zero_grad(set_to_none=True)
            d_loss = discriminator_loss(discriminator, style, generated, edge_fake, adversarial_criterion)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            generated = generator(photo)
            g_loss, parts = generator_loss(
                discriminator,
                photo,
                generated,
                adversarial_criterion,
                content_weight=args.content_weight,
                tv_weight=args.tv_weight,
            )
            g_loss.backward()
            optimizer_g.step()

            running["d_loss"] += float(d_loss.detach().cpu().item())
            running["g_loss"] += float(g_loss.detach().cpu().item())
            running["g_adv"] += parts["g_adv"]
            running["content"] += parts["content"]
            running["tv"] += parts["tv"]

            if args.log_every > 0 and (step % args.log_every == 0 or step == len(train_loader)):
                logger.info(
                    "epoch=%03d/%03d step=%04d/%04d d_loss=%.4f g_loss=%.4f "
                    "g_adv=%.4f content=%.4f tv=%.4f step_time=%.2fs",
                    epoch,
                    args.epochs,
                    step,
                    len(train_loader),
                    float(d_loss.detach().cpu().item()),
                    float(g_loss.detach().cpu().item()),
                    parts["g_adv"],
                    parts["content"],
                    parts["tv"],
                    time.time() - step_start,
                )

        epoch_log = {name: value / max(len(train_loader), 1) for name, value in running.items()}
        epoch_log["epoch"] = epoch
        epoch_log["epoch_time_sec"] = round(time.time() - epoch_start, 3)
        epoch_log["lr_g"] = optimizer_g.param_groups[0]["lr"]
        epoch_log["lr_d"] = optimizer_d.param_groups[0]["lr"]
        history.append(epoch_log)
        logger.info(
            "epoch=%03d summary d_loss=%.4f g_loss=%.4f g_adv=%.4f "
            "content=%.4f tv=%.4f time=%.2fs",
            epoch,
            epoch_log["d_loss"],
            epoch_log["g_loss"],
            epoch_log["g_adv"],
            epoch_log["content"],
            epoch_log["tv"],
            epoch_log["epoch_time_sec"],
        )

        save_validation_samples(generator, val_loader, device, sample_dir, epoch)
        save_json({"domain": args.domain, "history": history, "args": vars(args)}, run_dir / "history.json")
        save_history_csv(history, run_dir / "history.csv")
        logger.info("saved validation samples to %s", sample_dir / f"epoch_{epoch:03d}.jpg")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                args,
                history,
            )
            logger.info("saved checkpoint to %s", checkpoint_dir / f"epoch_{epoch:03d}.pth")
            save_checkpoint(
                run_dir / "last_model.pth",
                epoch,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                args,
                history,
            )
            logger.info("updated last checkpoint at %s", run_dir / "last_model.pth")


if __name__ == "__main__":
    main()
