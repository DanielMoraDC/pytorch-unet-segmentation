"""
Evaluate model on the test
"""

import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from unet_segmentation.data.dataset import SegmentationDataset
from unet_segmentation.metrics import iou

data_load_workers = 4
batch_size = 1
image_size = 512
ious_file = 'ious'

# ---------------
# Load model
# ---------------

unet = torch.load('unet_iter_1300000.pt').cuda()

# ---------------
# Load test dataset
# ---------------

test_dataset = SegmentationDataset('full_dataset',
                                   image_resize=image_size,
                                   subset='test')

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         num_workers=data_load_workers,
                         pin_memory=True)

# ---------------
# Show predictions
# ---------------

ious = []

for batch_idx, (imgs, masks) in enumerate(test_loader):

    ious += iou(masks.cuda(), unet(imgs.cuda()))
    if batch_idx % 250 == 0:
        batch_mean_iou = np.mean(
            ious[batch_idx*batch_size:(batch_idx+1)*batch_size]
        )
        print(f'Evaluated batch {batch_idx}: {batch_mean_iou:.2f}')

print(f'IOU on test is {np.mean(ious)} +- {np.std(ious)}')
np.save(ious_file, np.array(ious))
