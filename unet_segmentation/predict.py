"""
Predict model on image samples using CPU
"""

import torch

import numpy as np
import matplotlib.pyplot as plt

from unet_segmentation.data.dataset import SegmentationDataset

# ---------------
# Load model
# ---------------

unet = torch.load('unet.pt').cpu()

# ---------------
# Dataset
# ---------------

dataset = SegmentationDataset('seg_dataset', image_resize=256)

# ---------------
# Show predictions
# ---------------

n_rows = 2
fig, axs = plt.subplots(n_rows, 3, figsize=(7.0, n_rows * 3.0))

for i in range(n_rows):

    image, mask = dataset[i]
    img_size = image.shape[-1]
    predicted_mask = unet(torch.Tensor(image).unsqueeze_(0))
    predicted_mask = predicted_mask.detach().numpy()
    predicted_image = np.argmax(predicted_mask, axis=1)[0].astype('uint8')

    axs[i][0].imshow(np.transpose(image, (1, 2, 0)))
    axs[i][0].set_title('Sample image #{}'.format(i))
    axs[i][1].imshow(mask)
    axs[i][1].set_title('Sample mask #{}'.format(i))
    axs[i][2].imshow(predicted_image)
    axs[i][2].set_title('Predicted mask #{}'.format(i))

plt.tight_layout()
plt.show()
