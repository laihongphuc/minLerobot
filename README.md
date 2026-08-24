# minLerobot

A clean, minimal re-implementation of the core ideas behind Hugging Face’s [LeRobotDataset](https://github.com/huggingface/lerobot).

This project exists to help you **understand** the LeRobot dataset format by reading and experimenting with a simplified version.

Robot learning datasets are more complex than typical vision or language datasets: they are multi-modal (images + proprioception + actions), temporal / episodic, and high-frequency. LeRobot stores data in **file-size oriented** chunks (Parquet + MP4) and reconstructs **episode/frame oriented** access through metadata. This repo keeps that design and drops production complexity.

## What is implemented

- [x] Core dataset class with the same high-level spirit as `LeRobotDataset`
- [x] Loading of `meta/info.json`, episode metadata, and tabular data
- [x] Frame indexing (`dataset[idx]`)
- [x] `delta_timestamps` (temporal context windows), including padding near episode ends
- [x] Mean / std stats for state and action (`compute_norm_stats`)

Not implemented: dataset writing / recording, Hub download & caching, depth / multi-dataset / streaming, production-grade optimizations.

## Install

Use the `lerobot` conda env (recommended — it already has PyTorch, torchvision, and torchcodec):

```bash
git clone https://github.com/laihongphuc/minLerobot.git
cd minLerobot
conda activate lerobot
pip install -e ".[dev]"
```

You need a local dataset in LeRobot format (for example `so101_pick`):

```
<dataset>/
  meta/info.json
  data/chunk-000/file-000.parquet
  videos/observation.images.side/chunk-000/file-000.mp4
  ...
```

Copy `.env.example` to `.env` and set the path (`.env` is gitignored):

```bash
cp .env.example .env
# DATASET_PATH=/path/to/so101_pick
```

`python-dotenv` loads this automatically. You can still override it with `export DATASET_PATH=...`.

## Use the dataset

Load a single frame (state, action, and camera images):

```python
from minlerobot import MyLeRobotDataset

dataset = MyLeRobotDataset()  # uses DATASET_PATH from .env
print(len(dataset), dataset.fps, dataset.observation_image_keys)

item = dataset[0]
print(item["state"].shape)   # (6,)
print(item["action"].shape)  # (6,)
print(item["observation.images.side"].shape)  # (1, 3, H, W)
```

Use `delta_timestamp` to request a temporal window. Values are offsets in **seconds** relative to the current frame. Near episode boundaries, out-of-range steps are clamped and marked in `*_is_pad`.

```python
fps = dataset.fps
delta_timestamp = {
    "action": [i / fps for i in range(-10, 1)],            # 11 action steps
    "observation.images.side": [i / fps for i in range(8)],  # 8 image frames
}

dataset = MyLeRobotDataset(delta_timestamp=delta_timestamp)
item = dataset[307]
print(item["action"].shape)                  # (11, 6)
print(item["action_is_pad"].shape)           # (11,)
print(item["observation.images.side"].shape) # (8, 3, H, W)
```

Batch with the provided collate function:

```python
from torch.utils.data import DataLoader
from minlerobot import MyLeRobotDataset, collate_fn

loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
batch = next(iter(loader))
print(batch["state"].shape)
print(batch["action"].shape)
```

Optional image transforms (any `torchvision` transform that accepts a `(N, C, H, W)` tensor):

```python
from torchvision import transforms

dataset = MyLeRobotDataset(image_transform=transforms.Resize((224, 224)))
```

## Compute norm stats

Mean and std of `state` and `action` are written to JSON for later normalization.

CLI (after install):

```bash
minlerobot-norm-stats -o norm_stats.json
minlerobot-norm-stats -o norm_stats.json --device cpu --batch-size 64
```

Or from Python:

```python
from minlerobot import compute_norm_stats, save_norm_stats, load_norm_stats

stats = compute_norm_stats()  # uses DATASET_PATH from .env
save_norm_stats(stats, "norm_stats.json")

mean, std = load_norm_stats("norm_stats.json")["action"]
normalized_action = (action - mean) / std
```

Uses CUDA when it is available, otherwise CPU.

## Test

Tests expect a real dataset on disk. Put `DATASET_PATH` in `.env` first.

```bash
conda activate lerobot
pip install -e ".[dev]"
pytest tests/ -v
```

If the dataset path is missing, data-dependent tests are skipped. `test_norm_stats_roundtrip` still runs because it does not need videos.
