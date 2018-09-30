import torch
import numpy as np
import cv2

class RandomStretch(object):
    def __init__(self, max_stretch=0.05, interpolation='cubic'):
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, sample):
        scale = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        shape = np.array(sample.shape[:2]) * scale
        shape = tuple(shape.astype(np.int32))
        return cv2.resize(sample, shape, cv2.INTER_CUBIC)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        shape = sample.shape[:2]
        cy, cx = shape[0] // 2, shape[1] // 2
        y1, x1 = cy - self.size[0]//2, cx - self.size[1] // 2
        y2, x2 = cy + self.size[0]//2 + 1, cx + self.size[1] // 2 + 1
        return sample[y1:y2, x1:x2]

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        shape = sample.shape[:2]
        y1 = np.random.randint(0, shape[0] - self.size[0])
        x1 = np.random.randint(0, shape[1] - self.size[1])
        y2 = y1 + self.size[0]
        x2 = x1 + self.size[1]
        pad_right = pad_bottom = 0
        if y2 > shape[0]:
            pad_bottom = y2 - shape[0]
        if x2 > shape[1]:
            pad_right = x2 - shape[1]
        img_patch = sample[y1:y2, x1:x2]
        if pad_right or pad_bottom:
            img_patch = cv2.copyMakeBorder(img_patch, 0, pad_bottom, 0, pad_right,
                    cv2.BORDER_REPLICATE)
        return img_patch

class Normalize(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, sample):
        return (sample / 255. - self.mean) / self.std

class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))
