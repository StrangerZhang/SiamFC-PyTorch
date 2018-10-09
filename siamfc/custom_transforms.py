import torch
import numpy as np
import cv2

class RandomStretch(object):
    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, sample):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = (int(h * scale_h), int(w * scale_w))
        return cv2.resize(sample, shape, cv2.INTER_CUBIC)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        shape = sample.shape[:2]
        cy, cx = shape[0] // 2, shape[1] // 2
        ymin, xmin = cy - self.size[0]//2, cx - self.size[1] // 2
        ymax, xmax = cy + self.size[0]//2 + 1, cx + self.size[1] // 2 + 1
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right !=0 or top!=0 or bottom!=0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=0)
        return im_patch

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        shape = sample.shape[:2]
        y1 = np.random.randint(0, shape[0] - self.size[0])
        x1 = np.random.randint(0, shape[1] - self.size[1])
        y2 = y1 + self.size[0]
        x2 = x1 + self.size[1]
        img_patch = sample[y1:y2, x1:x2]
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
