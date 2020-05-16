import numpy as np
import torch
from torch.utils.data import DataLoader

from unet_segmentation.data.dataset import SegmentationDataset
from unet_segmentation.data.augmentation import DataAugmentation

from unet_segmentation.training.training import (
    TrainingParams,
    fit
)


# --------------------
# Training dataset
# --------------------

batch_size = 1
data_load_workers = 4
image_crop = 512
image_resize = 575

data_aug = DataAugmentation(
    central_crop_size=image_crop,
    h_flipping_chance=0.50,
    brightness_rate=0.10,
    contrast_rate=0.10,
    saturation_rate=0.10,
    hue_rate=0.05)

train_dataset = SegmentationDataset('full_dataset',
                                    image_resize=image_resize,
                                    data_augmentation=data_aug)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=data_load_workers,
                          pin_memory=True)

# --------------------
# Validation dataset
# --------------------

val_dataset = SegmentationDataset('full_dataset',
                                  subset='validation',
                                  image_resize=image_crop)

val_loader = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=data_load_workers,
                        pin_memory=True)

# --------------------
# Training
# --------------------


# Hard-code presence for each class
class_rate = np.array(
    # High value for background
    [0.50] +
    # Top,  Shorts,  Dress,  Skirt,  Trousers, Outwear
    [0.4039, 0.1173, 0.1587, 0.0988, 0.1774, 0.04485]
)

# The higher the presence, the lower the weight in the loss
base_weight = 0.25
class_loss_weight = torch.Tensor(class_rate.max() / class_rate * base_weight)
print(f'Using class weights: {class_loss_weight}')
cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_loss_weight)

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

params = TrainingParams(
    train_loader=train_loader,
    val_loader=val_loader,
    loss=cross_entropy_loss,
    lr=1e-3,
    momentum=0.99,
    stats_interval=25000,
    save_model_interval=25000,
    save_model_dir='.',
    n_epochs=100,
    n_classes=6 + 1,  # 6 + background
    device=device,
    checkpoint=None
    # checkpoint='unet_iter_750000.pt'
)

fit(params)
