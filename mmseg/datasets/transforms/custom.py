import random
import mmcv
import cv2
import albumentations as A
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


@TRANSFORMS.register_module()
class CLAHE2(BaseTransform):
    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8), prob=0.5):
        super().__init__()
        self.prob = prob
        self.clahe = A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results
        img = results['img']
        results['img'] = self.clahe(image=img)['image']
        return results


@TRANSFORMS.register_module()
class GaussianNoise(BaseTransform):
    def __init__(self, mean=0, std=1, prob=0.5):
        super().__init__()
        self.prob = prob
        self.mean = mean
        self.std = std

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results
        img = results['img']
        noise = np.random.normal(self.mean, self.std, img.shape) * 255
        noise = noise.clip(0, 255).astype(np.uint8)
        img = cv2.add(img, noise)
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class RandomGamma(BaseTransform):
    def __init__(self, gamma_range=(0.5, 1.5), prob=0.5):
        super().__init__()
        self.prob = prob
        self.gamma_range = gamma_range

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results
        img = results['img']
        gamma = random.uniform(*self.gamma_range)
        img = img.astype(np.float32) / 255.0
        img = np.power(img, gamma)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class CoarseDropout(BaseTransform):
    def __init__(self, num_holes_range,  hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0, fill_mask=0, p=0.5):
        super().__init__()
        self.dropout = A.CoarseDropout(num_holes_range=num_holes_range, hole_height_range=hole_height_range, hole_width_range=hole_width_range, fill_value=fill, mask_fill_value=fill_mask, p=p)

    def transform(self, results: dict) -> dict:
        img = results['img']
        mask = results['gt_seg_map']
        transformed = self.dropout(image=img, mask=mask)
        results['img'] = transformed['image']
        results['gt_seg_map'] = transformed['mask']
        return results


if __name__ == '__main__':
    # Test the custom transforms
    import random
    import os
    import mmcv
    from matplotlib import pyplot as plt
    imgs_path = '/workspace/data/pipeline/images/train'
    masks_path = '/workspace/data/pipeline/annotations/train'

    file_name = random.choice(os.listdir(imgs_path))
    print(file_name)
    img = mmcv.imread(os.path.join(imgs_path, file_name))
    mask = mmcv.imread(os.path.join(masks_path, file_name))
    results = dict(img=img, gt_semantic_seg=mask)

    # transform = GaussianNoise(mean=0.0, std=0.1, prob=1)
    transform = CoarseDropout(num_holes_range=(1, 3), p=1, hole_height_range=(0.1, 0.4), hole_width_range=(0.1, 0.4))
    # transform = CLAHE(prob=1)
    # transform = RandomGamma(gamma_range=(0.5, 1.5), prob=1)
    results = transform(results)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    print(np.max(results['gt_semantic_seg']))
    ax[1, 0].imshow(results['img'])
    ax[1, 1].imshow(results['gt_semantic_seg'] * 255)
    ax[0, 1].imshow(mask * 255)
    ax[0, 0].imshow(img)
    plt.savefig('transformed.png')