import os
import random
from glob import glob
from typing import List
from collections.abc import Iterable
from dataclasses import dataclass

import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


@dataclass
class DataAugmentation(object):

    central_crop_size: int = 512
    h_flipping_chance: float = 0.50
    brightness_rate: float = 0.10
    contrast_rate: float = 0.10
    saturation_rate: float = 0.10
    hue_rate: float = 0.05

    def augment(self, image, mask):
        # Color and illumination changes
        image = transforms.ColorJitter(brightness=self.brightness_rate,
                                       contrast=self.contrast_rate,
                                       saturation=self.saturation_rate,
                                       hue=self.hue_rate)(image)

        # Random crop the image
        random_crop = PairRandomCrop(image_size=self.central_crop_size)
        image, mask = random_crop(image, mask)

        # Random horizontal flipping
        if random.random() > self.h_flipping_chance:
            image, mask = TF.hflip(image), TF.hflip(mask)

        return image, mask


class DeepFashion2Dataset(Dataset):

    def __init__(self,
                 path: str,
                 image_resize: int,
                 data_augmentation: DataAugmentation = None):
        self._image_resize = image_resize
        self._data_augmentation = data_augmentation

        # Collect all images
        images_path = os.path.join(path, 'images')
        images = glob(os.path.join(images_path, '*'))

        masks_path = os.path.join(path, 'masks')

        def _mask_path(img_path: str) -> str:
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            return os.path.join(masks_path, image_id + '.png')

        # Find image - mask pairs
        self._image_mask_pairs = np.array([
            (img_path, _mask_path(img_path))
            for img_path in images
            if os.path.isfile(_mask_path(img_path))
        ])
        print(f'Read {len(images)} images, built ' +
              f'{len(self._image_mask_pairs)} image pairs')

    def _transform(self, image, mask):
        # Resizing + Data augmentation
        resize = SkimageResize(size=(self._image_resize, self._image_resize))
        image, mask = resize(image), resize(mask)

        if self._data_augmentation is not None:
            image, mask = self._data_augmentation.augment(image, mask)

        # Image normalization
        image, mask = np.array(image), np.array(mask)
        image = transforms.ToTensor()(image)  # Implicit NCHW conversion
        image = transforms.Normalize(mean=(0,), std=(1,))(image)
        return image, mask

    def __len__(self):
        return len(self._image_mask_pairs)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, mask_path = self._image_mask_pairs[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        return self._transform(image, mask)


class SkimageResize(object):

    """ Resizes numpy array preserving aspect ratio """

    def __init__(self, **args):
        self._transform_fn = transforms.Resize
        self._args = args

    def __call__(self, img):
        img_arr = np.array(img)
        return self._transform_fn(**self._args)(Image.fromarray(img_arr))


class PairRandomCrop(object):

    def __init__(self, image_size: int):
        self._image_size = image_size

    def __call__(self, image, mask):
        # Compute paddings for random crop given dimensions
        crop_params = transforms.RandomCrop.get_params(
            image,
            output_size=(self._image_size, self._image_size)
        )
        start_y, start_x, new_height, new_width = crop_params
        # Apply crop given computed paddings
        image = TF.crop(image, start_y, start_x, new_height, new_width)
        mask = TF.crop(mask, start_y, start_x, new_height, new_width)
        return image, mask


if __name__ == '__main__':

    augmentation = DataAugmentation(
        central_crop_size=512,
        h_flipping_chance=0.50,
        brightness_rate=0.10,
        contrast_rate=0.10,
        saturation_rate=0.10,
        hue_rate=0.05
    )

    data = DeepFashion2Dataset('seg_dataset',
                               image_resize=575,
                               data_augmentation=augmentation)

    n_rows = 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(7.0, n_rows*3.0))

    for i in range(n_rows):
        image, mask = data[i]
        axs[i][0].imshow(np.transpose(image, (1, 2, 0)))
        axs[i][0].set_title('Sample image #{}'.format(i))
        axs[i][1].imshow(mask)
        axs[i][1].set_title('Sample mask #{}'.format(i))

    plt.show()
