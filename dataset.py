import os 
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchcodec.decoders import VideoDecoder
import json



class MyLeRobotDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ):
        super().__init__()
        self.data_path = data_path 
        self.metadata = self._process_metadata()
        self.chunk_index, self.state, self.action, self.episode_index, \
            self.frame_index, self.timestamps, self.file_index  = self._process_all_data()
     
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
    
    def _index_to_file_path(
        self,
        index: int,
    ):
        """Return the file path for the given index"""
        chunk_id = self.chunk_index[index]
        file_id = self.file_index[index]
        return f"chunk-{chunk_id:03d}/file-{file_id:03d}.mp4"
    
    def _index_to_timestamp(
        self, 
        index: int,
    ):
        """Return the timestamp for the given index"""
        return self.timestamps[index]

    # data processing
    def _process_all_data(
        self,
    ):
        """
        Process all the data from the chunks and concatenate the state and action values into a single numpy array
        """
        chunk_ids_list = os.listdir(os.path.join(self.data_path, "data"))
        chunk_ids_list = sorted(chunk_ids_list, key=lambda x: int(x.split("-")[1]))
        state = []
        action = []
        episode_index = []
        frame_index = []
        timestamps = []
        chunk_index = []
        file_index = []
        for chunk_ids in chunk_ids_list:
            current_state, current_action , current_episode_index, current_frame_index, \
                current_timestamps, current_file_index = self._process_chunk_data(chunk_path=os.path.join(self.data_path, "data", chunk_ids))
            state.append(current_state)
            action.append(current_action)
            episode_index.append(current_episode_index)
            frame_index.append(current_frame_index)    
            timestamps.append(current_timestamps)
            chunk_index.append(np.full(current_state.shape[0], int(chunk_ids.split("-")[1])))
            file_index.append(current_file_index)
        chunk_index = np.concatenate(chunk_index, axis=0)
        state = np.concatenate(state, axis=0)
        action = np.concatenate(action, axis=0)
        episode_index = np.concatenate(episode_index, axis=0)
        frame_index = np.concatenate(frame_index, axis=0)
        timestamps = np.concatenate(timestamps, axis=0)
        file_index = np.concatenate(file_index, axis=0)
        assert state.shape[0] == action.shape[0], "state and action must have the same length"
        assert state.shape[0] == self.num_frames, "state and action must have the same length as the number of frames"
        return chunk_index, state, action, episode_index, frame_index, timestamps, file_index

    def _process_chunk_data(
        self,
        chunk_path: str,
    ):
        """
        Process the data from the chunk index, and extract action and state values, episode index, frame index, and timestamps
        """
        parquet_file_list = [file for file in os.listdir(chunk_path) if file.endswith(".parquet")]
        parquet_file_list = sorted(parquet_file_list, key=lambda x: int(x.split(".")[0].split("-")[1]))
        file_index = []
        state = []
        action = []
        episode_index = []
        frame_index = [] # frame index in the episode
        timestamps = [] # timestamps of the frames

        for file in parquet_file_list:
            df = pd.read_parquet(os.path.join(chunk_path, file))
            assert "action" in df.columns, "action column not found in the parquet file"
            assert "observation.state" in df.columns, "observation.state column not found in the parquet file"
            # robot-related data
            current_state = np.stack(df["observation.state"].to_numpy())
            current_action = np.stack(df["action"].to_numpy())
            state.append(current_state)
            action.append(current_action)
            # episode-related data
            current_episode_index = df["episode_index"].to_numpy()
            current_frame_index = df["frame_index"].to_numpy()
            current_timestamps = df["timestamp"].to_numpy()
            episode_index.append(current_episode_index)
            frame_index.append(current_frame_index)
            timestamps.append(current_timestamps)
            # file-related data
            current_file_ids = int(file.split(".")[0].split("-")[1])
            file_index.append(np.full(current_state.shape[0], current_file_ids))
        state = np.concatenate(state, axis=0)
        action = np.concatenate(action, axis=0)
        episode_index = np.concatenate(episode_index, axis=0)
        frame_index = np.concatenate(frame_index, axis=0)
        timestamps = np.concatenate(timestamps, axis=0)
        file_index = np.concatenate(file_index, axis=0)
        return state, action, episode_index, frame_index, timestamps, file_index
    
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        result = {
            "state": torch.from_numpy(self.state[index]),
            "action": torch.from_numpy(self.action[index]),
            "episode_index": self.episode_index[index],
            "frame_index": self.frame_index[index],
            "timestamp": self.timestamps[index],
        }
        file_path = self._index_to_file_path(index)

        for key in self.observation_image_keys:
            video_path = os.path.join(self.data_path, "videos", key, file_path)
            video = VideoDecoder(video_path)
            frame = video[index]
            result[key] = frame
        return result

def collate_fn(batch_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for the dataset"""
    result: Dict[str, torch.Tensor] = {
        "state": torch.stack([item["state"] for item in batch_samples], dim=0),
        "action": torch.stack([item["action"] for item in batch_samples], dim=0),
    }
    for key in batch_samples[0].keys():
        if "observation" not in key:
            continue
        result[key] = torch.stack([item[key] for item in batch_samples], dim=0)
    return result