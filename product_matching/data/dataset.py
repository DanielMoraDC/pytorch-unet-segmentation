import os
import random
import json
from typing import Tuple

import skimage.io

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from product_matching.data.augmentation import DataAugmentation
from product_matching.data.bbox import BoundingBox


class ProductMatchingDataset(Dataset):

    def __init__(self,
                 images_path: str,
                 path: str,
                 image_resize: int,
                 image_mean: np.ndarray = [0.485, 0.456, 0.406],
                 image_std: np.array = [0.229, 0.224, 0.225],
                 data_augmentation: DataAugmentation = None):
        """

        Args:
            images_path: Path to directory where images are stored.
            path: Path to the json file.
            image_resize: Size images will be resized into.
            data_augmentation: Data augmentation parameters.
        """
        self._images_path = images_path
        self._path = path
        self._image_resize = image_resize
        self._image_mean = image_mean
        self._image_std = image_std
        self._data_augmentation = data_augmentation

        with open(path, 'r') as file:
            self._data = json.load(file)

        assert self._data['category_to_pairs'].keys() == self._data['pairs_to_data'].keys()  # noqa

        self._pairs_per_category = {
            category: len(values)
            for category, values in self._data['category_to_pairs'].items()
        }
        self._categories = list(self._data['category_to_pairs'].keys())
        self._pairs_per_epoch = min(self._pairs_per_category.values())

    def _transform(self, example):
        image = skimage.io.imread(
            os.path.join(self._images_path, example['image_path'])
        )

        # Read bounding box
        bbox = BoundingBox(*example['bounding_box'])
        if self._data_augmentation is not None:
            bbox = self._data_augmentation.augment_bounding_box(image, bbox)

        # Crop image around bounding box
        product_image = bbox.crop_image(image)
        product_image = TF.to_pil_image(product_image)

        # Augment image, if needed
        if self._data_augmentation is not None:
            product_image = self._data_augmentation.augment_image(product_image)

        # Image resize
        product_image = transforms.Resize(
            (self._image_resize, self._image_resize),
            interpolation=Image.NEAREST
        )(product_image)

        # Image normalization using generic pre-trained means and std
        # See https://pytorch.org/docs/stable/torchvision/models.html
        product_image = transforms.ToTensor()(product_image)
        return transforms.Normalize(mean=self._image_mean,
                                    std=self._image_std)(product_image)

    def __len__(self):
        return len(self._categories) * self._pairs_per_epoch

    def _idx_to_category(self, idx):
        return self._categories[idx % len(self._categories)]

    def _sample_triplet(self, category: str) -> Tuple[dict, dict, dict]:
        anchor_id, negative_id = random.sample(
            self._data['category_to_pairs'][category],
            2
        )
        assert anchor_id != negative_id

        anchor, positive = random.sample(
            self._data['pairs_to_data'][category][str(anchor_id)],
            2
        )
        negative = random.sample(
            self._data['pairs_to_data'][category][str(negative_id)],
            1
        )[0]
        return anchor, positive, negative

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        category = self._idx_to_category(idx)
        anchor, positive, negative = self._sample_triplet(category)
        return (self._transform(anchor), \
                self._transform(positive), \
                self._transform(negative)), 0  # Return a fake label


if __name__ == '__main__':

    # import random
    # import matplotlib.pyplot as plt
    #
    # def _display(image, tag, ax):
    #     ax.imshow(np.transpose(image, (1, 2, 0)))
    #     ax.set_title(tag)
    #
    # dataset = ProductMatchingDataset(
    #     images_path=os.path.join('datasets', 'dataset'),
    #     path=os.path.join('datasets', 'product_matching_dataset', 'train.json'),
    #     image_resize=256,
    #     data_augmentation=DataAugmentation()
    # )
    #
    # n_rows = 5
    # idxs = [random.randint(0, len(dataset)) for _ in range(n_rows)]
    # fig, axs = plt.subplots(n_rows, 3, figsize=(7.0, n_rows*3.0))
    #
    # for i, idx in enumerate(idxs):
    #     data, _ = dataset[idx]
    #     anchor, positive, negative = data
    #     _display(anchor, 'anchor', axs[i][0])
    #     _display(positive, 'positive', axs[i][1])
    #     _display(negative, 'negative', axs[i][2])
    #
    # plt.tight_layout()
    # plt.show()

    import random
    import matplotlib.pyplot as plt

    train_data = ProductMatchingDataset(
        images_path=os.path.join('datasets', 'dataset'),
        path=os.path.join('datasets', 'product_matching_dataset_toy', 'train.json'),
        image_resize=256,
        image_mean=[0, 0, 0],
        image_std=[1, 1, 1],
        data_augmentation=DataAugmentation()
    )

    def _display(image, tag, ax):
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title(tag)

    n_rows = len(train_data)
    fig, axs = plt.subplots(n_rows, 3, figsize=(7.0, n_rows * 3.0))

    for i in range(len(train_data)):
        data, _ = train_data[i]
        anchor, positive, negative = data
        _display(anchor, 'anchor', axs[i][0])
        _display(positive, 'positive', axs[i][1])
        _display(negative, 'negative', axs[i][2])

    plt.tight_layout()
    plt.show()

