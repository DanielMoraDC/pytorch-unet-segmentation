from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class BoundingBox(object):

    x1: int
    y1: int
    x2: int
    y2: int

    def random_augment(self,
                       image: np.ndarray,
                       max_pad_rate: float):
        """
        Provides a new bounding box which is a result of randomly
        increasing/decreasing each side of the box independently.

        Given any of the box sides "s", the new side length is computed as:

            s' = s + Uniform(-max_pad_rate * s, max_pad_rate * s)

        And the new width is computed accordingly.

        :param image: Reference image bounding box belongs to.
        :param max_pad_rate: Maximum modification of any the bounding box sides
            with respect to the current side length.
        :return: Modified bounding box.
        """
        height, width = image.shape[:2]
        box_height, box_width = self.y2 - self.y1, self.x2 - self.x1
        max_y_diff = int(box_height * max_pad_rate)
        max_x_diff = int(box_width * max_pad_rate)

        def _clamp_y(y: int) -> int:
            return max(0, min(y, height - 1))

        def _clamp_x(x: int) -> int:
            return max(0, min(x, width - 1))

        def _sample_y_diff() -> int:
            return int(np.random.uniform(-max_y_diff, max_y_diff))

        def _sample_x_diff() -> int:
            return int(np.random.uniform(-max_x_diff, max_x_diff))

        return BoundingBox(
            y1=_clamp_y(_sample_y_diff() + self.y1),
            y2=_clamp_y(_sample_y_diff() + self.y2),
            x1=_clamp_x(_sample_x_diff() + self.x1),
            x2=_clamp_x(_sample_x_diff() + self.x2),
        )

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        return image[self.y1:self.y2, self.x1:self.x2]

    def display(self, ax) -> None:
        rect = patches.Rectangle((self.x1, self.y1),
                                 self.x2 - self.x1, self.y2 - self.y1,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)


if __name__ == '__main__':
    import json
    import random
    import os
    import matplotlib.pyplot as plt
    import skimage.io

    dataset_json_path = os.path.join('datasets',
                                     'product_matching_dataset',
                                     'test.json')
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)

    # Sample random instance from data
    category = random.choice(list(data['category_to_pairs'].keys()))
    example = data['pairs_to_data'][category][
        str(random.choice(data['category_to_pairs'][category]))
    ][0]

    # Compare augmented bounding box from original
    image = skimage.io.imread(os.path.join('datasets',
                                           'dataset',
                                           example['image_path']))
    bbox = BoundingBox(*example['bounding_box'])
    augmented_bbox = bbox.random_augment(image)

    fig, axs = plt.subplots(2, 2)
    axs[0][0].imshow(image)
    bbox.display(axs[0][0])
    axs[0][0].set_title('Original')
    axs[1][0].imshow(bbox.crop_image(image))
    axs[1][0].set_title('Original cropped')

    axs[0][1].imshow(image)
    augmented_bbox.display(axs[0][1])
    axs[0][1].set_title('Augmented')
    axs[1][1].imshow(augmented_bbox.crop_image(image))
    axs[1][1].set_title('Augmented cropped')

    plt.tight_layout()
    plt.show()
