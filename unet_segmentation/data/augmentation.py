import random
from dataclasses import dataclass

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
