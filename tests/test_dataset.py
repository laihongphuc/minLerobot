"""Pytest cases for MyLeRobotDataset.

Run from the repo root after `pip install -e .[dev]`:
    pytest tests/ -v
"""

import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from minlerobot import (
    MyLeRobotDataset,
    collate_fn,
    get_dataset_path,
    load_norm_stats,
    save_norm_stats,
)

try:
    DATA_PATH = get_dataset_path()
except ValueError:
    DATA_PATH = ""

print(f"DATA_PATH: {DATA_PATH}")
FPS = 30
ACTION_DIM = 6
SAMPLE_INDICES = [0, 1000, 5000]
NEAR_EPISODE_BOUNDARY_INDEX = 307


def _dataset_available() -> bool:
    return os.path.isdir(os.path.join(DATA_PATH, "meta"))


@pytest.fixture(scope="module")
def dataset():
    if not _dataset_available():
        pytest.skip(f"dataset not found at {DATA_PATH}")
    return MyLeRobotDataset(data_path=DATA_PATH)


@pytest.fixture(scope="module")
def delta_timestamps():
    return {
        "action": [i / FPS for i in range(-10, 1)],
        "observation.images.side": [i / FPS for i in range(8)],
    }


@pytest.fixture(scope="module")
def delta_dataset(delta_timestamps):
    if not _dataset_available():
        pytest.skip(f"dataset not found at {DATA_PATH}")
    return MyLeRobotDataset(data_path=DATA_PATH, delta_timestamp=delta_timestamps)


def test_process_data(dataset):
    n_frames = dataset.num_frames
    assert dataset.state.shape == (n_frames, ACTION_DIM)
    assert dataset.action.shape == (n_frames, ACTION_DIM)
    assert dataset.episode_index.shape[0] == n_frames
    assert dataset.frame_index.shape[0] == n_frames
    assert dataset.timestamps.shape[0] == n_frames


def test_metadata(dataset):
    assert dataset.num_frames == dataset.metadata["total_frames"]
    assert dataset.fps == dataset.metadata["fps"]
    assert "features" in dataset.metadata
    expected_image_keys = [
        key
        for key in dataset.metadata["features"]
        if key.startswith("observation.images")
    ]
    assert dataset.observation_image_keys == expected_image_keys
    for key, shape in dataset.frame_size.items():
        assert list(shape) == dataset.metadata["features"][key]["shape"]


def test_dataset(dataset):
    assert len(dataset) == dataset.num_frames

    item = dataset[0]
    assert item["state"].shape == (ACTION_DIM,)
    assert item["action"].shape == (ACTION_DIM,)
    assert item["episode_index"] == dataset.episode_index[0]
    assert item["frame_index"] == dataset.frame_index[0]
    assert item["timestamp"] == dataset.timestamps[0]
    torch.testing.assert_close(item["state"], torch.from_numpy(dataset.state[0]))
    torch.testing.assert_close(item["action"], torch.from_numpy(dataset.action[0]))

    for key in dataset.observation_image_keys:
        height, width, channels = dataset.frame_size[key]
        image = item[key]
        assert image.ndim >= 3
        assert image.shape[-3:] == (channels, height, width)

    for index in SAMPLE_INDICES:
        sample = dataset[index]
        assert sample["state"].shape == (ACTION_DIM,)
        assert sample["action"].shape == (ACTION_DIM,)
        for key in dataset.observation_image_keys:
            height, width, channels = dataset.frame_size[key]
            assert sample[key].shape[-3:] == (channels, height, width)


def test_dataloader(dataset):
    batch_size = 32
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    batch = next(iter(dataloader))
    assert batch["state"].shape == (batch_size, ACTION_DIM)
    assert batch["action"].shape == (batch_size, ACTION_DIM)
    for key in dataset.observation_image_keys:
        height, width, channels = dataset.frame_size[key]
        assert batch[key].shape[0] == batch_size
        assert batch[key].shape[-3:] == (channels, height, width)


def test_image_transform(dataset):
    resized = MyLeRobotDataset(
        data_path=DATA_PATH,
        image_transform=transforms.Resize((224, 224)),
    )
    item = resized[0]
    for key in dataset.observation_image_keys:
        assert item[key].shape[-2:] == (224, 224)

    for index in SAMPLE_INDICES:
        sample = resized[index]
        for key in dataset.observation_image_keys:
            assert sample[key].shape[-2:] == (224, 224)


def test_norm_stats_roundtrip(tmp_path):
    stats = {
        "state": (torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)),
        "action": (torch.ones(ACTION_DIM), torch.full((ACTION_DIM,), 0.5)),
    }
    json_path = tmp_path / "norm_stats.json"
    save_norm_stats(stats, json_path)
    loaded = load_norm_stats(json_path)
    assert set(loaded) == set(stats)
    for key in stats:
        torch.testing.assert_close(loaded[key][0], stats[key][0])
        torch.testing.assert_close(loaded[key][1], stats[key][1])


def test_query_indices(delta_dataset):
    # episodes 1: [0,302], 2: [303, ...]
    abs_idx = NEAR_EPISODE_BOUNDARY_INDEX
    query_indices, padding = delta_dataset._get_query_indices(abs_idx)

    assert padding["action"] == [True] * 6 + [False] * 5
    assert padding["observation.images.side"] == [False] * 8
    assert set(query_indices) == set(delta_dataset.delta_indices)


def test_act_dataset(delta_dataset):
    item = delta_dataset[NEAR_EPISODE_BOUNDARY_INDEX]

    assert item["action"].shape == (11, ACTION_DIM)
    assert item["state"].shape == (ACTION_DIM,)
    assert item["action_is_pad"].shape == (11,)
    assert item["action_is_pad"].dtype == torch.bool
    assert item["observation.images.side"].shape[0] == 8
    assert item["observation.images.side_is_pad"].shape == (8,)

    height, width, channels = delta_dataset.frame_size["observation.images.side"]
    assert item["observation.images.side"].shape[-3:] == (channels, height, width)


def test_act_dataloader(delta_dataset):
    batch_size = 32
    dataloader = DataLoader(
        delta_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    batch = next(iter(dataloader))

    assert batch["state"].shape == (batch_size, ACTION_DIM)
    assert batch["action"].shape == (batch_size, 11, ACTION_DIM)
    assert batch["observation.images.side"].shape[0] == batch_size
    assert batch["observation.images.side"].shape[1] == 8
    height, width, channels = delta_dataset.frame_size["observation.images.side"]
    assert batch["observation.images.side"].shape[-3:] == (channels, height, width)
    assert batch["observation.images.side_is_pad"].shape == (batch_size, 8)
