from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from .dataset import Flickr8kCaptionDataset
from .model import MiniBlip2Config, MiniBlip2ForCaptioning
from .utils import ensure_dir, get_device, save_trainable_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mini-BLIP2 on Flickr8k captions.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--limit-images", type=int, default=200)
    parser.add_argument("--captions-per-image", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--language-model", default="facebook/opt-125m")
    parser.add_argument("--num-query-tokens", type=int, default=16)
    parser.add_argument("--qformer-hidden-size", type=int, default=256)
    parser.add_argument("--qformer-layers", type=int, default=2)
    return parser.parse_args()


def build_collate_fn(image_processor, tokenizer, max_length: int):
    def collate_fn(batch: list[dict]) -> dict:
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
        tokens = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = tokens.input_ids.clone()
        labels[tokens.attention_mask == 0] = -100
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": labels,
            "captions": captions,
        }

    return collate_fn


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model)

    dataset = Flickr8kCaptionDataset(
        data_root=args.data_root,
        limit_images=args.limit_images,
        captions_per_image=args.captions_per_image,
    )
    train_size = max(1, int(len(dataset) * 0.9))
    val_size = len(dataset) - train_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset, val_dataset = dataset, None

    collate_fn = build_collate_fn(image_processor, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    config = MiniBlip2Config(
        vision_model_name=args.vision_model,
        language_model_name=args.language_model,
        num_query_tokens=args.num_query_tokens,
        qformer_hidden_size=args.qformer_hidden_size,
        qformer_layers=args.qformer_layers,
    )
    model = MiniBlip2ForCaptioning(config).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)

    log_path = output_dir / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss\n", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        pixel_values=batch["pixel_values"].to(device),
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    val_loss += outputs.loss.item()
            val_loss /= max(1, len(val_loader))

        log_line = f"{epoch},{train_loss:.6f},{val_loss:.6f}\n"
        with log_path.open("a", encoding="utf-8") as file:
            file.write(log_line)
        print(log_line.strip())

        save_trainable_checkpoint(model, output_dir / f"mini_blip2_epoch_{epoch}.pt")

    save_trainable_checkpoint(model, output_dir / "mini_blip2_latest.pt")


if __name__ == "__main__":
    main()
