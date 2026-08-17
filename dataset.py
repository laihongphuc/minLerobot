import os 
from typing import Tuple, Dict, List
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
        self.metadata = self._process_metadata()
        self.chunk_list, self.state, self.action = self._process_all_data()
     
    # metadata processing
    def _process_metadata(
        self,
    ):
        metadata_path = os.path.join(self.data_path, "meta") 
        # read info.json
        with open(os.path.join(metadata_path, "info.json"), "r") as f:
            info = json.load(f)
        
        return info
    
    @property 
    def observation_image_keys(
        self,
    ):
        """Return a list of image keys in observation"""
        return [x for x in self.metadata['features'].keys() if x.startswith("observation.images")]

    @property
    def frame_size(
        self,
    ):
        """Return a dictionary of frame size for images in observation - up, side, front"""
        result = {x: self.metadata['features'][x]['shape'] for x in self.observation_image_keys}
        return result

    @property
    def num_frames(
        self,
    ):
        return self.metadata["total_frames"]

    @property
    def fps(
        self,
    ):
        return self.metadata["fps"]

    # data processing
    def _process_all_data(
        self,
    ):
        """
        Process all the data from the chunks and concatenate the state and action values into a single numpy array
        """
        chunk_list = os.listdir(os.path.join(self.data_path, "data"))
        chunk_list = sorted(chunk_list, key=lambda x: int(x.split("-")[1]))
        state = []
        action = []
        for chunk_index in chunk_list:
            current_state, current_action = self._process_chunk_data(chunk_path=os.path.join(self.data_path, "data", chunk_index))
            state.append(current_state)
            action.append(current_action)
        state = np.concatenate(state, axis=0)
        action = np.concatenate(action, axis=0)
        assert state.shape[0] == action.shape[0], "state and action must have the same length"
        assert state.shape[0] == self.num_frames, "state and action must have the same length as the number of frames"
        return chunk_list, state, action

    def _process_chunk_data(
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
    
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        result = {
            "state": torch.from_numpy(self.state[index]),
            "action": torch.from_numpy(self.action[index]),
        }
        for key in self.observation_image_keys:
            frame_size = self.frame_size[key]
            result[key] = torch.zeros(frame_size)
        return result

def collate_fn(batch_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for the dataset"""
    result: Dict[str, torch.Tensor] = {
        "state": torch.stack([item["state"] for item in batch_samples], dim=0),
        "action": torch.stack([item["action"] for item in batch_samples], dim=0),
    }
    for key in batch_samples[0].keys():
        if key == "state" or key == "action":
            continue
        result[key] = torch.stack([item[key] for item in batch_samples], dim=0)
    return result