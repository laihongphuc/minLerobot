import os 
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json


class FakeVideoDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ):
        super().__init__()
        self.data_path = data_path 

    def _process_metadata(
        self,
    ):
        metadata_path = os.path.join(self.data_path, "meta") 
        # read info.json
        with open(os.path.join(metadata_path, "info.json"), "r") as f:
            info = json.load(f)
        self.observation = {"state": [], "image": []}
        self.action = {}
        # 2. Load observation state 
        self.observation = {"state": [], "image": []}
        # 3. Load observation image
        return info

    def _process_data(
        self,
        chunk_path: str,
    ):
        """
        Process the data from the chunk index, and extract action and state values
        """
        parquet_file_list = [file for file in os.listdir(chunk_path) if file.endswith(".parquet")]
        parquet_file_list = sorted(parquet_file_list, key=lambda x: int(x.split(".")[0].split("-")[1]))
        state = []
        action = []
        for file in parquet_file_list:
            df = pd.read_parquet(os.path.join(chunk_path, file))
            assert "action" in df.columns, "action column not found in the parquet file"
            assert "observation.state" in df.columns, "observation.state column not found in the parquet file"
            current_state = np.stack(df["observation.state"].to_numpy())
            current_action = np.stack(df["action"].to_numpy())
            state.append(current_state)
            action.append(current_action)
        return np.concatenate(state), np.concatenate(action)

