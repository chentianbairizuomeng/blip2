from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from .dataset import read_flickr8k_examples
from .model import MiniBlip2Config, MiniBlip2ForCaptioning
from .utils import get_device, load_trainable_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate captions with trained Mini-BLIP2.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default="outputs/mini_blip2_latest.pt")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--language-model", default="facebook/opt-125m")
    parser.add_argument("--num-query-tokens", type=int, default=16)
    parser.add_argument("--qformer-hidden-size", type=int, default=256)
    parser.add_argument("--qformer-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model)

    config = MiniBlip2Config(
        vision_model_name=args.vision_model,
        language_model_name=args.language_model,
        num_query_tokens=args.num_query_tokens,
        qformer_hidden_size=args.qformer_hidden_size,
        qformer_layers=args.qformer_layers,
    )
    model = MiniBlip2ForCaptioning(config).to(device)
    load_trainable_checkpoint(model, args.checkpoint, device)
    model.eval()

    examples = read_flickr8k_examples(args.data_root, limit_images=200)[: args.num_samples]
    for idx, example in enumerate(examples, start=1):
        image = Image.open(example.image_path).convert("RGB")
        pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"[{idx}] image: {Path(example.image_path).name}")
        print(f"gold: {example.caption}")
        print(f"pred: {generated}")
        print()


if __name__ == "__main__":
    main()
