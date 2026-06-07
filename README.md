# Mini-BLIP2 Reproduction

This repository contains a lightweight BLIP-2 style image captioning reproduction for the course assignment.

## What Is Implemented

- Frozen vision encoder: `openai/clip-vit-base-patch32`
- Trainable bridge: custom Mini Q-Former
- Trainable projection layer into language embedding space
- Frozen language decoder: `facebook/opt-125m`
- Flickr8k loader that reads the first 200 images and captions
- Training and caption generation scripts

## Repository Layout

```text
blip2/
├── code/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
├── data/
├── report/
├── index.html
├── requirements.md
└── requirements.txt
```

## Manual Steps You Need To Do

### 1. Download Flickr8k

Download the dataset from:

https://www.kaggle.com/datasets/adityajn105/flickr8k

Then extract it into `data/`. The loader supports common layouts such as:

```text
data/
  Images/
  captions.txt
```

or:

```text
data/
  Flickr8k_Dataset/
  Flickr8k.token.txt
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 3. Verify the dataset loader

```powershell
python -m code.dataset --data-root data --limit-images 200
```

### 4. Train the model

```powershell
python -m code.train --data-root data --output-dir outputs --epochs 3 --batch-size 2
```

### 5. Generate example captions

```powershell
python -m code.generate --data-root data --checkpoint outputs/mini_blip2_latest.pt --num-samples 5
```

## Entire + Git Workflow

Entire has been enabled in this repository. To keep the assignment evidence complete:

1. Work in small steps.
2. Commit after each module.
3. Keep the Entire-linked conversation process.
4. Push your commits and checkpoint metadata after important milestones.

Useful commands:

```powershell
git log --oneline
git push
```

## Current Commit Milestones

At this stage the repository already includes these small commits:

- `chore: enable Entire tracking for Codex`
- `feat: load Flickr8k first 200 images and captions`
- `feat: implement Mini Q-Former bridge model`
- `feat: add training and caption generation scripts`

## Notes

- Only the Mini Q-Former and projection layer are trainable.
- The vision encoder and language decoder are frozen to match the assignment idea.
- The final report should include training logs, generated captions, Entire conversation evidence, and Git commit history.
