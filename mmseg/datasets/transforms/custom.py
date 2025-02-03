import random
import mmcv
import cv2
import numpy as np
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results


@TRANSFORMS.register_module()
class GaussianBlur(BaseTransform):
    def __init__(self, sigma: float, prob: float = 0.5):
        super().__init__()
        self.prob = prob
        self.sigma = sigma

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results
        img = results['img']
        results['img'] = cv2.GaussianBlur(img, (0, 0), self.sigma)
        return results
    

@TRANSFORMS.register_module()
class RGBShift(BaseTransform):
    def __init__(self, r_shift=(0, 0), g_shift=(0, 0), b_shift=(0, 0), prob=0.5):
        super().__init__()
        self.prob = prob
        self.r_shift = r_shift
        self.g_shift = g_shift
        self.b_shift = b_shift

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results
        img = results['img']
        r_shift = random.uniform(*self.r_shift)
        g_shift = random.uniform(*self.g_shift)
        b_shift = random.uniform(*self.b_shift)
        img = img.astype(np.float32)
        img[..., 0] += r_shift
        img[..., 1] += g_shift
        img[..., 2] += b_shift
        img -= np.mean([r_shift, g_shift, b_shift])
        img = np.clip(img, 0, 255).astype(np.uint8)
        results['img'] = img
        return results
