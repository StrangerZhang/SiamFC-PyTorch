import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
from torch.utils.data.dataset import Dataset

from .config import config

class ImagnetVIDDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]:x[1] for x in self.meta_data}
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False)
        self.num = len(self.video_names) if config.num_per_epoch is None or not training\
                else config.num_per_epoch

    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def __getitem__(self, idx):
        idx = idx % len(self.video_names)
        video = self.video_names[idx]
        trajs = self.meta_data[video]
        # sample one trajs
        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]
        assert len(traj) > 1, "video_name: {}".format(video)
        # sample exemplar
        exemplar_idx = np.random.choice(list(range(len(traj))))
        exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx]+".{:02d}.x.jpg".format(trkid))
        exemplar_img = self.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        # sample instance
        low_idx = max(0, exemplar_idx - config.frame_range)
        up_idx = min(len(traj), exemplar_idx + config.frame_range)

        # create sample weight, if the sample are far away from center
        # the probability being choosen are high
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx+1:up_idx], p=weights)
        instance_name = os.path.join(self.data_dir, video, instance+".{:02d}.x.jpg".format(trkid))
        instance_img = self.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        if np.random.rand(1) < config.gray_ratio:
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
        exemplar_img = self.z_transforms(exemplar_img)
        instance_img = self.x_transforms(instance_img)
        return exemplar_img, instance_img

    def __len__(self):
        return self.num
