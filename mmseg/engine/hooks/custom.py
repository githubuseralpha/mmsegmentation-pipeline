import os
import random

from mmengine.hooks import Hook
from mmseg.registry import HOOKS
import mmcv
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

@HOOKS.register_module()
class SavePredictionHook(Hook):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.mean = torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std = torch.tensor([58.395, 57.12, 57.375]).cuda()

    @staticmethod
    def resize(img, mask, size, pad_val=0):
        h, w = img.shape[:2]
        if h > w:
            new_h, new_w = size, int(w * size / h)
        else:
            new_h, new_w = int(h * size / w), size
        pad_h = size - new_h
        pad_w = size - new_w
        img = mmcv.imresize(img, (new_w, new_h), interpolation='nearest')
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=pad_val)
        mask = mmcv.imresize(mask, (new_w, new_h), interpolation='nearest')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_val)
        return img, mask
    
    @staticmethod
    def _calculate_IOU(mask, gt, category_id=1):
        mask = mask == category_id
        gt = gt == category_id
        intersection = np.logical_and(mask, gt).sum()
        union = np.logical_or(mask, gt).sum()
        return intersection / union

    @staticmethod
    def standardize(img, mean, std):
        img = img.permute(1, 2, 0)
        img = img - mean
        img = img / std
        img = img.permute(2, 0, 1)
        return img

    def calculate_mean_std(self, train_dataloader):
        # Initialize with double precision for numerical stability
        mean = torch.zeros(3, dtype=torch.float64)
        m2 = torch.zeros(3, dtype=torch.float64)  # for variance calculation
        total_pixels = 0
        
        # Auto-detect device
        device = 'cuda'
        mean = mean.to(device)
        m2 = m2.to(device)

        for data in tqdm(train_dataloader.dataset):
            inputs = data['inputs'].to(device)
            
            # Convert to float32 and normalize to [0, 1] range first
            # This prevents overflow in squared calculations
            inputs = inputs.float() / 255.0
            
            _, h, w = inputs.shape
            n_pixels = h * w
            
            # Welford's online algorithm for numerical stability
            channel_sum = torch.sum(inputs, dim=[1, 2])
            channel_sum_sq = torch.sum(inputs ** 2, dim=[1, 2])
            
            delta = channel_sum - mean * n_pixels
            mean += delta / (total_pixels + n_pixels)
            m2 += channel_sum_sq - (delta ** 2) / (total_pixels + n_pixels)
            
            total_pixels += n_pixels
        
        # Final calculation with epsilon to prevent sqrt of negative numbers
        variance = m2 / (total_pixels - 1)  # sample variance
        std = torch.sqrt(variance + 1e-8)  # add small epsilon
        
        # Scale back to original 0-255 range
        mean_0_255 = mean * 255
        std_0_255 = std * 255
        print(mean_0_255, std_0_255)
        
        return mean_0_255.cpu().numpy(), std_0_255.cpu().numpy()

    def before_val_epoch(self, runner, *args, **kwargs):
        print('Saving predictions...')
        if self.mean is None or self.std is None:
            self.mean, self.std = self.calculate_mean_std(runner.train_dataloader)

        dataloader = runner.val_dataloader
        model = runner.model
        
        os.makedirs(self.output_dir, exist_ok=True)
        for i, data in tqdm(enumerate(dataloader)):
            filename = data['data_samples'][0].img_path
            filename = os.path.basename(filename)
            img = data['inputs'][0]
            # img = img.flip(dims=(0,))
            gt = data['data_samples'][0].gt_sem_seg.data
            with torch.no_grad():
                img = self.standardize(img.float().cuda(), self.mean, self.std).unsqueeze(0)
                pred = model(img)
            pred = pred.cpu().numpy().squeeze()
            gt = gt.cpu().numpy().squeeze()
            pred = np.argmax(pred, axis=0)

            gt = mmcv.imresize(gt, pred.shape[::-1], interpolation='nearest')

            iou = self._calculate_IOU(pred, gt)
            with open(os.path.join(self.output_dir, f'{filename[:-4]}_iou.txt'), 'w') as f:
                f.write(str(iou))
