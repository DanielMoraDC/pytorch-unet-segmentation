import random
from dataclasses import dataclass

import numpy as np

from torchvision import transforms
import torchvision.transforms.functional as TF

from product_matching.data.bbox import BoundingBox


@dataclass
class DataAugmentation(object):

    bounding_box_pad_rate: float = 0.125

    h_flipping_chance: float = 0.50
    brightness_rate: float = 0.05
    contrast_rate: float = 0.05
    saturation_rate: float = 0.05
    hue_rate: float = 0.05

    def augment_bounding_box(self,
                             image: np.ndarray,
                             bounding_box: BoundingBox):
        return bounding_box.random_augment(
            image,
            max_pad_rate=self.bounding_box_pad_rate
        )

    def augment_image(self, image):
        # Color and illumination changes
        image = transforms.ColorJitter(brightness=self.brightness_rate,
                                       contrast=self.contrast_rate,
                                       saturation=self.saturation_rate,
                                       hue=self.hue_rate)(image)

        # Random horizontal flipping
        if random.random() > self.h_flipping_chance:
            image = TF.hflip(image)

        return image
