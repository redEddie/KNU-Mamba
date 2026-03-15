import logging
import random
import pickle

import cv2
import h5py
import os
import torch
import numpy as np
from MambaVLA.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)


class LiberoDataset():
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 32,
            action_dim: int = 7,
            state_dim: int = 45,
            max_len_data: int = 136,
            chunck_size: int = 1,
            start_idx: int = 0,
            demos_per_task: int = 1,
    ):
        self.data_directory = data_directory
        # Always keep dataset tensors on CPU to avoid CUDA tensors in DataLoader workers
        # self.device = "cpu"
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_len_data = max_len_data
        self.chunck_size = chunck_size
        self.start_idx = start_idx
        self.demos_per_task = demos_per_task

        self.data_dir = sim_framework_path(self.data_directory)
        logging.info("The dataset is loading from {}".format(self.data_dir))  # show the dataset directory

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.data_directory = data_directory

        benchmark_type = os.path.basename(data_directory)
        task_emb_dir = sim_framework_path("language_embeddings")

        with open(task_emb_dir + "/" + benchmark_type + ".pkl", 'rb') as f:
            tasks = pickle.load(f)

        data_embs = []
        actions = []
        masks = []
        self.dataset_metadata = []
        all_states = []

        file_list = os.listdir(self.data_dir)

        for file in file_list:
            if not file.endswith('.hdf5'):
                continue

            filename = os.path.basename(file).split('.')[0][:-5]
            task_emb = tasks[filename]
            file_path = os.path.join(self.data_dir, file)

            f = h5py.File(file_path, 'r')

            log.info("Loading metadata for demo: {}".format(file))

            demo_keys_list = list(f["data"].keys())
            indices = np.argsort([int(elem[5:]) for elem in demo_keys_list])

            for i in indices[start_idx: start_idx + demos_per_task]:
                demo_name = demo_keys_list[i]
                demo = f["data"][demo_name]
                demo_length = demo.attrs["num_samples"]

                zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                action_data = demo['actions'][:]

                zero_actions[0, :demo_length, :] = action_data
                zero_mask[0, :demo_length] = 1

                joint_states = demo['obs']['joint_states'][:]
                gripper_states = demo['obs']['gripper_states'][:]
                robot_states = np.concatenate((joint_states, gripper_states), axis=-1)

                actions.append(zero_actions)
                masks.append(zero_mask)
                all_states.append(robot_states)
                data_embs.append(task_emb)
                
                # Store metadata for lazy loading instead of the actual images
                self.dataset_metadata.append({
                    "file_path": file_path,
                    "demo_name": demo_name
                })

            f.close()

        self.actions = torch.from_numpy(np.concatenate(actions)).float()

        self.all_states = all_states
        self.data_embs = data_embs
        self.tasks = tasks
        self.masks = torch.from_numpy(np.concatenate(masks)).float()

        self.num_data = len(self.dataset_metadata)

        self.num_data = len(self.dataset_metadata)

        self.slices = self.get_slices()

    def get_slices(self):  #Extract sample slices that meet certain conditions
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.chunck_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.chunck_size}")
            else:
                slices += [
                    (i, start, start + self.chunck_size) for start in range(T - self.chunck_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        Notice: this reads directly from HDF5 lazily. Should be used sparingly or completely removed if not needed.
        """
        print("Warning: get_all_observations reads all data from disk synchronously.")
        result = []
        for i in range(self.num_data):
            T = int(self.masks[i].sum().item())
            f = h5py.File(self.dataset_metadata[i]["file_path"], 'r')
            agentview_chunk = f["data"][self.dataset_metadata[i]["demo_name"]]['obs']['agentview_rgb'][:T]
            result.append(agentview_chunk)
            f.close()
            
        return torch.from_numpy(np.concatenate(result, axis=0))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        obs = {}

        task_emb = self.data_embs[i]

        robot_states = self.all_states[i][start:start+1]

        # Keep on CPU; device transfer handled in training loop
        task_emb = task_emb.float() if isinstance(task_emb, torch.Tensor) else torch.tensor(task_emb, dtype=torch.float32)

        # Lazy load image arrays directly from HDF5 based on metadata
        meta = self.dataset_metadata[i]
        with h5py.File(meta["file_path"], 'r') as f:
            demo_data = f["data"][meta["demo_name"]]['obs']
            agentview_rgb = demo_data['agentview_rgb'][start:start+1]
            eye_in_hand_rgb = demo_data['eye_in_hand_rgb'][start:start+1]

        agentview_rgb = torch.from_numpy(agentview_rgb).float().permute(0, 3, 1, 2) / 255.
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb).float().permute(0, 3, 1, 2) / 255.

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        obs["agentview_image"] = agentview_rgb
        obs["eye_in_hand_image"] = eye_in_hand_rgb
        obs["lang_emb"] = task_emb

        obs["robot_states"] = torch.from_numpy(robot_states).float()

        return obs, act, mask
