from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CaptionExample:
    image_path: Path
    caption: str


def find_caption_file(data_root: Path) -> Path:
    candidates = [
        data_root / "captions.txt",
        data_root / "Flickr8k.token.txt",
        data_root / "flickr8k" / "captions.txt",
        data_root / "archive" / "captions.txt",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(data_root.rglob("captions.txt")) + sorted(data_root.rglob("Flickr8k.token.txt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not find captions.txt or Flickr8k.token.txt. "
        "Place the Kaggle Flickr8k files under data/."
    )


def find_image_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "Images",
        data_root / "images",
        data_root / "Flicker8k_Dataset",
        data_root / "Flickr8k_Dataset",
        data_root / "flickr8k" / "Images",
        data_root / "archive" / "Images",
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*")):
            return path

    dirs_with_images: list[Path] = []
    for path in data_root.rglob("*"):
        if path.is_dir() and any(child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir()):
            dirs_with_images.append(path)
    if dirs_with_images:
        return sorted(dirs_with_images, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError("Could not find a Flickr8k image directory under data/.")


def _parse_caption_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.lower().startswith("image,caption"):
        return None

    if "\t" in line:
        image_id, caption = line.split("\t", 1)
        image_name = image_id.split("#", 1)[0]
        return image_name.strip(), caption.strip()

    if "," in line:
        image_name, caption = line.split(",", 1)
        return image_name.strip(), caption.strip().strip('"')

    return None


def read_flickr8k_examples(
    data_root: str | Path,
    limit_images: int = 200,
    captions_per_image: int = 1,
) -> list[CaptionExample]:
    data_root = Path(data_root)
    caption_file = find_caption_file(data_root)
    image_dir = find_image_dir(data_root)

    image_to_captions: dict[str, list[str]] = {}
    for line in caption_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_caption_line(line)
        if parsed is None:
            continue
        image_name, caption = parsed
        image_to_captions.setdefault(image_name, []).append(caption)

    examples: list[CaptionExample] = []
    for image_name in sorted(image_to_captions)[:limit_images]:
        image_path = image_dir / image_name
        if not image_path.exists():
            continue
        for caption in image_to_captions[image_name][:captions_per_image]:
            examples.append(CaptionExample(image_path=image_path, caption=caption))

    if not examples:
        raise RuntimeError(
            f"No usable examples found. Caption file: {caption_file}; image dir: {image_dir}"
        )
    return examples


class Flickr8kCaptionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        limit_images: int = 200,
        captions_per_image: int = 1,
        image_transform: Callable | None = None,
    ) -> None:
        self.examples = read_flickr8k_examples(data_root, limit_images, captions_per_image)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        image = Image.open(example.image_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return {
            "image": image,
            "caption": example.caption,
            "image_path": str(example.image_path),
        }


def describe_examples(examples: Iterable[CaptionExample], max_rows: int = 5) -> str:
    lines = []
    for idx, example in enumerate(examples):
        if idx >= max_rows:
            break
        lines.append(f"{idx + 1}. {example.image_path.name}: {example.caption}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Flickr8k caption samples.")
    parser.add_argument("--data-root", default="data", help="Path to the Flickr8k data directory.")
    parser.add_argument("--limit-images", type=int, default=200)
    args = parser.parse_args()

    examples = read_flickr8k_examples(args.data_root, limit_images=args.limit_images)
    print(f"Loaded {len(examples)} caption examples from first {args.limit_images} images.")
    print(describe_examples(examples))


if __name__ == "__main__":
    main()
