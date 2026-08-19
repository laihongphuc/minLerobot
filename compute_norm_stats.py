from typing import Tuple
import torch 
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dataset import MyLeRobotDataset, collate_fn
import json


class RunningStats:
    def __init__(self):
        self._mean: torch.Tensor = None 
        self._mean_of_squares: torch.Tensor = None 
        self.count: int = 0
    
    def update(self, batch: torch.Tensor): 
        assert batch.ndim == 2, "Batch must be a 2D tensor"
        num_elements, vector_length = batch.shape
        if self.count == 0  :
            self._mean = batch.mean(dim=0)
            self._mean_of_squares = batch.pow(2).mean(dim=0)
        else:
            self._mean = (self._mean * self.count + batch.mean(dim=0) * num_elements) / (self.count + num_elements)
            self._mean_of_squares = (self._mean_of_squares * self.count + batch.pow(2).mean(dim=0) * num_elements) / (self.count + num_elements)
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def compute_norm_stats(dataset_path: str) -> dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the norm stats of the dataset
    
    Returns:
        dict: The norm stats (mean, std) of the action and state
    """
    dataset = MyLeRobotDataset(dataset_path)
    dataloader = create_dataloader(dataset)

    keys = ["state", "action"]
    stats = {key: RunningStats() for key in keys}
    for batch in tqdm(dataloader, desc="Computing norm stats"):
        for key in keys:
            batch[key] = batch[key].to("cuda")
            stats[key].update(batch[key])
    return {key: (stats[key].mean, stats[key].std) for key in keys}

def save_norm_stats(stats: dict[str, Tuple[torch.Tensor, torch.Tensor]], json_path: str):
    """Save the norm stats to a json file"""
    stats = {key: {"mean": stats[key][0].tolist(), "std": stats[key][1].tolist()} for key in stats}
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

def load_norm_stats(json_path: str) -> dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Load the norm stats from a json file"""
    with open(json_path, "r") as f:
        stats = json.load(f)
    return {key: (torch.tensor(stats[key]["mean"]), torch.tensor(stats[key]["std"])) for key in stats}

