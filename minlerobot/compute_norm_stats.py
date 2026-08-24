import argparse
import json
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minlerobot.dataset import MyLeRobotDataset, collate_fn


class RunningStats:
    def __init__(self):
        self._mean: torch.Tensor = None
        self._mean_of_squares: torch.Tensor = None
        self.count: int = 0

    def update(self, batch: torch.Tensor):
        assert batch.ndim == 2, "Batch must be a 2D tensor"
        num_elements, _vector_length = batch.shape
        if self.count == 0:
            self._mean = batch.mean(dim=0)
            self._mean_of_squares = batch.pow(2).mean(dim=0)
        else:
            self._mean = (
                self._mean * self.count + batch.mean(dim=0) * num_elements
            ) / (self.count + num_elements)
            self._mean_of_squares = (
                self._mean_of_squares * self.count
                + batch.pow(2).mean(dim=0) * num_elements
            ) / (self.count + num_elements)
        self.count += num_elements

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        variance = self._mean_of_squares - self.mean.pow(2)
        return torch.sqrt(torch.clamp(variance, min=1e-6))

    def reset(self):
        self._mean = None
        self._mean_of_squares = None
        self.count = 0

    def __str__(self):
        return f"Mean: {self.mean}, Std: {self.std}"


def create_dataloader(
    dataset: MyLeRobotDataset,
    batch_size: int = 128,
    shuffle: bool = False,
):
    """Create a dataloader for the dataset"""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def compute_norm_stats(
    dataset_path: str,
    device: Optional[str] = None,
    batch_size: int = 128,
) -> dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the norm stats of the dataset

    Returns:
        dict: The norm stats (mean, std) of the action and state
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MyLeRobotDataset(dataset_path)
    dataloader = create_dataloader(dataset, batch_size=batch_size)

    keys = ["state", "action"]
    stats = {key: RunningStats() for key in keys}
    for batch in tqdm(dataloader, desc="Computing norm stats"):
        for key in keys:
            stats[key].update(batch[key].to(device))
    return {key: (stats[key].mean.cpu(), stats[key].std.cpu()) for key in keys}


def save_norm_stats(
    stats: dict[str, Tuple[torch.Tensor, torch.Tensor]], json_path: str
):
    """Save the norm stats to a json file"""
    stats = {
        key: {"mean": stats[key][0].tolist(), "std": stats[key][1].tolist()}
        for key in stats
    }
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(json_path: str) -> dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Load the norm stats from a json file"""
    with open(json_path, "r") as f:
        stats = json.load(f)
    return {
        key: (torch.tensor(stats[key]["mean"]), torch.tensor(stats[key]["std"]))
        for key in stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/std of state and action for a LeRobot-format dataset."
    )
    parser.add_argument("dataset_path", help="Path to a local LeRobot dataset directory")
    parser.add_argument(
        "-o",
        "--output",
        default="norm_stats.json",
        help="Where to write the JSON stats (default: norm_stats.json)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda or cpu (default: cuda if available, else cpu)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    stats = compute_norm_stats(
        args.dataset_path, device=args.device, batch_size=args.batch_size
    )
    save_norm_stats(stats, args.output)
    print(f"Wrote norm stats to {args.output}")
    for key, (mean, std) in stats.items():
        print(f"{key}: mean={mean.tolist()}, std={std.tolist()}")


if __name__ == "__main__":
    main()
