import torch
import cv2
import os
import numpy as np
import pickle
from torch.utils.data.dataset import Dataset

from .config import config

class ImagnetVIDDataset(Dataset):
    def __init__(self, video_names, data_dir, z_transforms, x_transforms):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]:x[1] for x in self.meta_data}
        # self.center = config.instance_size // 2
        # self.size = config.exemplar_size // 2
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

    def __getitem__(self, idx):
        idx = idx // config.sample_pairs_per_video
        video = self.video_names[idx]
        trajs = self.meta_data[video]
        # sample one trajs
        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]
        assert len(traj) > 1, "video_name: {}".format(video)
        # sample exemplar
        exemplar_idx = np.random.choice(list(range(len(traj))))
        exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx]+".{:02d}.x.JPEG".format(trkid))
        exemplar_img = cv2.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        low_idx = max(0, exemplar_idx - config.frame_range)
        up_idx = min(len(traj), exemplar_idx + config.frame_range)
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx+1:up_idx])
        instance_name = os.path.join(self.data_dir, video, instance+".{:02d}.x.JPEG".format(trkid))
        instance_img = cv2.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        exemplar_img = self.z_transforms(exemplar_img)
        instance_img = self.x_transforms(instance_img)
        return exemplar_img, instance_img

    def __len__(self):
        return len(self.video_names * config.sample_pairs_per_video)


