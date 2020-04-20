import numpy as np
import pandas as pd

from typing import List

import skimage.io

from PIL import (
    Image,
    ImageDraw
)


def get_mask(height: int,
             width: int,
             polygons: List,
             category_id: int) -> np.ndarray:
    default_value = 0
    # See https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    mask = Image.new(mode='L', size=(width, height), color=default_value)
    # Fill each of the input polygons
    for polygon in polygons:
        ImageDraw.Draw(mask).polygon(polygon,
                                     outline=category_id,
                                     fill=category_id)
    return np.array(mask)


