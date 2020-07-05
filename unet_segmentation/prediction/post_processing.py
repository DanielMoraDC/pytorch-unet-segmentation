from dataclasses import dataclass
from typing import List

from PIL import Image

import torch
import numpy as np
import skimage.transform
import skimage.morphology
from scipy.signal import medfilt
from skimage.measure import regionprops


@dataclass
class PredictedClass(object):
    area_ratio: float
    class_name: str
    class_id: int


@dataclass
class PredictedMask(PredictedClass):

    binary_mask: np.ndarray

    @staticmethod
    def from_prediction(binary_mask: np.ndarray,
                        prediction: PredictedClass):
        return PredictedMask(
            binary_mask=binary_mask,
            area_ratio=prediction.area_ratio,
            class_name=prediction.class_name,
            class_id=prediction.class_id
        )

    def resize(self, img: Image):
        width, height = img.size
        new_binary_mask = skimage.transform.resize(
            self.binary_mask,
            (height, width),
            anti_aliasing=False,
            order=0,  # mode=nearest
            preserve_range=True).astype('uint8')

        return PredictedMask(
            binary_mask=new_binary_mask,
            area_ratio=self.area_ratio,
            class_name=self.class_name,
            class_id=self.class_id
        )


def prediction_to_classes(prediction_img: np.ndarray,
                          class_id_to_name: dict) -> List[PredictedClass]:
    predicted_classes = [
        class_id
        for class_id in np.unique(prediction_img)
        if class_id != 0
    ]
    img_area = prediction_img.shape[0] * prediction_img.shape[1]

    return [
        PredictedClass(
            area_ratio=(prediction_img == class_id).sum() / img_area,
            class_name=class_id_to_name[class_id],
            class_id=class_id
        )
        for class_id in predicted_classes
    ]


def remove_predictions(predictions: List[PredictedClass],
                       img: np.ndarray) -> np.ndarray:
    result = img.copy()
    for prediction in predictions:
        result[result == prediction.class_id] = 0
    return result


def mask_from_prediction(prediction: PredictedClass,
                         img: np.ndarray,
                         smoothing_kernel_size: int = 7) -> PredictedMask:
    # Create binary mask for detection
    binary_img = np.zeros((img.shape))
    class_idxs = np.where(img == prediction.class_id)
    binary_img[class_idxs] = 1

    # Closing to remove spurious pixels
    closed_img = skimage.morphology.closing(binary_img)
    # Smooth edges with median filtering
    smoothed_img = medfilt(closed_img, kernel_size=smoothing_kernel_size)
    return PredictedMask.from_prediction(
        binary_mask=smoothed_img,
        prediction=prediction
    )
