import json
import math
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class MyLeRobotDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_transform: Optional[Callable] = None,
        delta_timestamp: Optional[dict[str, list[float]]] = None,
        time_tolerance: float = 0.04,
    ):
        super().__init__()
        self.data_path = data_path
        self.metadata = self._process_metadata()
        (
            self.chunk_index,
            self.state,
            self.action,
            self.episode_index,
            self.frame_index,
            self.timestamps,
            self.file_index,
        ) = self._process_all_data()
        self.image_transform = image_transform
        self.time_tolerance = time_tolerance
        self.delta_timestamp = delta_timestamp or {}
        self.delta_indices: dict[str, list[int]] = {}

        # Action chunk related
        _timestamp_keys = ["action", "state"]
        _timestamp_keys.extend(self.observation_image_keys)
        if delta_timestamp is not None:
            # Only support timestamp keys in the list _timestamp_keys
            for key in delta_timestamp:
                if key not in _timestamp_keys:
                    raise ValueError(f"Invalid key: {key} for delta timestamp")

            assert self.check_delta_timestamps(
                self.delta_timestamp, self.fps, self.time_tolerance
            ), "delta timestamps are not valid"
            self.delta_indices = self.get_delta_indices(self.delta_timestamp, self.fps)

    def _process_metadata(
        self,
    ):
        metadata_path = os.path.join(self.data_path, "meta")
        with open(os.path.join(metadata_path, "info.json"), "r") as f:
            info = json.load(f)

        return info

    @property
    def observation_image_keys(
        self,
    ):
        """Return a list of image keys in observation"""
        return [
            x
            for x in self.metadata["features"].keys()
            if x.startswith("observation.images")
        ]

    @property
    def frame_size(
        self,
    ):
        """Return a dictionary of frame size for images in observation - up, side, front"""
        result = {
            x: self.metadata["features"][x]["shape"]
            for x in self.observation_image_keys
        }
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

    def check_delta_timestamps(
        self, delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float
    ) -> bool:
        """Why we need to check the delta timestamps?
        The delta timestamps are used to compute the chunk index => need to be valid
        """
        for key in delta_timestamps:
            eval_tensor = [
                math.fabs(round(ts * fps) - ts * fps) < tolerance_s
                for ts in delta_timestamps[key]
            ]
            if not all(eval_tensor):
                return False
        return True

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

    def get_delta_indices(
        self,
        delta_timestamps: dict[str, list[float]],
        fps: int,
    ) -> dict[str, list[int]]:
        """Return the delta indices for the given delta timestamps"""
        return {
            key: [round(ts * fps) for ts in delta_timestamps[key]]
            for key in delta_timestamps
        }

    @staticmethod
    def _clamp(value: Any, min_value: Any, max_value: Any) -> Any:
        """Clamp the value between the min and max values"""
        return max(min(value, max_value), min_value)

    def _get_query_indices(
        self, abs_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices with delta timestamps

        Implementation explaination: For the near end or near start of the episode,
        we choose to pad the value with the first or last value of the episode.
        """
        current_ep = self.episode_index[abs_idx]
        query_indices = {}
        padding = {}
        for key in self.delta_indices:
            padding[key] = [False for _ in range(len(self.delta_indices[key]))]
            query_indices[key] = []
            current_min, current_max = None, None
            for i, rel_idx in enumerate(self.delta_indices[key]):
                current_idx = rel_idx + abs_idx
                if current_idx < 0:
                    padding[key][i] = True
                elif current_idx >= len(self):
                    padding[key][i] = True
                else:
                    if current_ep == self.episode_index[current_idx]:
                        if current_min is None or current_min > current_idx:
                            current_min = current_idx
                        if current_max is None or current_max < current_idx:
                            current_max = current_idx
                    else:
                        padding[key][i] = True
                query_indices[key].append(current_idx)
            query_indices[key] = [
                self._clamp(current_idx, current_min, current_max)
                for current_idx in query_indices[key]
            ]
        return query_indices, padding

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
            (
                current_state,
                current_action,
                current_episode_index,
                current_frame_index,
                current_timestamps,
                current_file_index,
            ) = self._process_chunk_data(
                chunk_path=os.path.join(self.data_path, "data", chunk_ids)
            )
            state.append(current_state)
            action.append(current_action)
            episode_index.append(current_episode_index)
            frame_index.append(current_frame_index)
            timestamps.append(current_timestamps)
            chunk_index.append(
                np.full(current_state.shape[0], int(chunk_ids.split("-")[1]))
            )
            file_index.append(current_file_index)
        chunk_index = np.concatenate(chunk_index, axis=0)
        state = np.concatenate(state, axis=0)
        action = np.concatenate(action, axis=0)
        episode_index = np.concatenate(episode_index, axis=0)
        frame_index = np.concatenate(frame_index, axis=0)
        timestamps = np.concatenate(timestamps, axis=0)
        file_index = np.concatenate(file_index, axis=0)
        assert (
            state.shape[0] == action.shape[0]
        ), "state and action must have the same length"
        assert (
            state.shape[0] == self.num_frames
        ), "state and action must have the same length as the number of frames"
        return (
            chunk_index,
            state,
            action,
            episode_index,
            frame_index,
            timestamps,
            file_index,
        )

    def _process_chunk_data(
        self,
        chunk_path: str,
    ):
        """
        Process the data from the chunk index, and extract action and state values, episode index, frame index, and timestamps
        """
        parquet_file_list = [
            file for file in os.listdir(chunk_path) if file.endswith(".parquet")
        ]
        parquet_file_list = sorted(
            parquet_file_list, key=lambda x: int(x.split(".")[0].split("-")[1])
        )
        file_index = []
        state = []
        action = []
        episode_index = []
        frame_index = []  # frame index in the episode
        timestamps = []  # timestamps of the frames

        for file in parquet_file_list:
            df = pd.read_parquet(os.path.join(chunk_path, file))
            assert "action" in df.columns, "action column not found in the parquet file"
            assert (
                "observation.state" in df.columns
            ), "observation.state column not found in the parquet file"
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
        query_indices, padding = self._get_query_indices(index)
        result = {
            "episode_index": self.episode_index[index],
            "frame_index": self.frame_index[index],
            "timestamp": self.timestamps[index],
        }
        file_path = self._index_to_file_path(index)

        # Process
        for key in ["action", "state"]:
            if key not in query_indices:
                _indices_list = [index]
                _padding = torch.tensor([False])
            else:
                _indices_list = query_indices[key]
                _padding = torch.tensor(padding[key])
            source = self.action if key == "action" else self.state
            result[key] = torch.from_numpy(source[_indices_list]).squeeze(0)
            result[f"{key}_is_pad"] = _padding

        for key in self.observation_image_keys:
            video_path = os.path.join(self.data_path, "videos", key, file_path)
            video = VideoDecoder(video_path)
            if key not in query_indices:
                _indices_list = [index]
                _padding = torch.tensor([False])
            else:
                _indices_list = query_indices[key]
                _padding = torch.tensor(padding[key])
            frame = video.get_frames_at(indices=[int(i) for i in _indices_list]).data
            if self.image_transform is not None:
                frame = self.image_transform(frame)
            result[key] = frame
            result[f"{key}_is_pad"] = _padding
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
