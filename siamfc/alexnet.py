import torch
import numpy as np
import torch.nn.functional as F

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from .config import config

"""
class SiameseAlexNetRaw(nn.Module):
    def __init__(self):
        super(SiameseAlexNetRaw, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1)
        )
        self.exemplar = None

    def load_weights(self):
        self.load_state_dict(alexnet(pretrained=True).state_dict(), strict=False)

    def forward(self, x):
        exemplar, instance = x
        if exemplar is not None and instance is not None:
            batch_size = exemplar.shape[0]
            exemplar = self.features(exemplar)
            instance = self.features(instance)
            score_map = []
            if batch_size > 1:
                for i in range(batch_size):
                    score_map.append(F.conv2d(instance[i:i+1], exemplar[i:i+1]))
                return torch.cat(score_map, dim=0)
            else:
                return F.conv2d(instance, exemplar)
        elif exemplar is not None and instance is None:
            self.exemplar = self.features(exemplar)
        else:
            instance = self.features(instance)
            response_map = F.conv2d(instance, self.exemplar)
            return response_map
"""
class SiameseAlexNet(nn.Module):
    def __init__(self, gpu_id, train=True):
        super(SiameseAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )
        self.corr_bias = nn.Parameter(torch.zeros(1))
        if train:
            gt, weight = self._create_gt_mask((config.train_response_sz, config.train_response_sz))
            with torch.cuda.device(gpu_id):
                self.train_gt = torch.from_numpy(gt).cuda()
                self.train_weight = torch.from_numpy(weight).cuda()
            gt, weight = self._create_gt_mask((config.response_sz, config.response_sz))
            with torch.cuda.device(gpu_id):
                self.valid_gt = torch.from_numpy(gt).cuda()
                self.valid_weight = torch.from_numpy(weight).cuda()
        self.exemplar = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        exemplar, instance = x
        if exemplar is not None and instance is not None:
            batch_size = exemplar.shape[0]
            exemplar = self.features(exemplar)
            instance = self.features(instance)
            score_map = []
            N, C, H, W = instance.shape
            if N > 1:
                for i in range(N):
                    score = F.conv2d(instance[i:i+1], exemplar[i:i+1]) * config.response_scale + self.corr_bias
                    score_map.append(score)
                return torch.cat(score_map, dim=0)
            else:
                return F.conv2d(instance, exemplar) * config.response_scale + self.bias
        elif exemplar is not None and instance is None:
            # inference used
            self.exemplar = self.features(exemplar)
        else:
            # inference used we don't need to scale the reponse or add bias
            instance = self.features(instance)
            score_map = []
            for i in range(instance.shape[0]):
                score_map.append(F.conv2d(instance[i:i+1], self.exemplar))
            return torch.cat(score_map, dim=0)

    def loss(self, pred):
        return F.binary_cross_entropy_with_logits(pred, self.gt)

    def weighted_loss(self, pred):
        if self.training:
            return F.binary_cross_entropy_with_logits(pred, self.train_gt,
                    self.train_weight, reduction='sum') / config.train_batch_size # normalize the batch_size
        else:
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
                    self.valid_weight, reduction='sum') / config.train_batch_size # normalize the batch_size

    def _create_gt_mask(self, shape):
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        return mask.astype(np.float32), weights.astype(np.float32)
