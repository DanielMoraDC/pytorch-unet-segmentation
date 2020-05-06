import os
from dataclasses import dataclass

from unet import (
    Unet,
    initialize_weights
)

from metrics import iou

from data import (
    DeepFashion2Dataset,
    DataAugmentation
)

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision


DATA_PATH = 'seg_dataset'

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False


# Evaluation metrics: https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation


# ---------------
# Parameters
# ---------------

@dataclass
class Evaluation(object):

    iteration: int

    images: np.ndarray
    masks: np.ndarray
    predictions: np.ndarray

    loss_value: float
    iou_value: float

    def track(self, writer: SummaryWriter) -> None:
        writer.add_images('inputs', self.images, self.iteration)
        writer.add_images('groundtruth', self.masks, self.iteration)
        writer.add_images('predictions', self.predictions,self.iteration)
        writer.add_scalar('loss', self.loss_value, self.iteration)
        writer.add_scalar('iou', self.iou_value, iteration)


def _validation_eval(model: Unet,
                     loss_fn: callable,
                     validation_loader: DataLoader,
                     iteration: int,
                     n_samples: int = 5) -> Evaluation:

    losses, ious = [], []
    image_samples, mask_samples, predictions = [], [], []
    sample_idxs = np.random.randint(0, len(validation_loader), size=n_samples)

    with torch.no_grad():

        for i, (imgs, masks) in enumerate(validation_loader):

            batch_imgs = imgs.to(device).float()
            batch_masks = masks.to(device).long()

            predicted_masks = model(batch_imgs)
            loss = loss_fn(predicted_masks, batch_masks).item()

            # Compute metrics
            iou_val = iou(batch_masks, predicted_masks).item()

            losses.append(loss)
            ious.append(iou_val)

            if i in sample_idxs:
                image_samples.append(imgs)
                mask_samples.append(masks)
                predictions.append(predicted_masks)

    return Evaluation(iteration=iteration,
                      images=torch.cat(image_samples, dim=0),
                      masks=torch.unsqueeze(torch.cat(mask_samples, dim=0),
                                            dim=1),
                      predictions=torch.unsqueeze(
                          torch.argmax(torch.cat(predictions, dim=0), dim=1),
                          dim=1
                      ),
                      loss_value=loss,
                      iou_value=iou_val)


load_from = None
# load_from = 'unet_iter_300000.pt'
n_epochs = 500
batch_size = 1
n_channels = 3
n_classes = 6 + 1  # background
data_load_workers = 4
lr = 0.001
tensorboard_dir = 'logs'
stats_interval = 5
save_model_interval = 50000
image_crop = 512
image_resize = 575

# Create weights for weighted loss
class_rate = np.array(
    # High value for background
    [0.50] +

    # Top,  Shorts,  Dress,  Skirt,  Trousers, Outwear
    [0.4039, 0.1173, 0.1587, 0.0988, 0.1774, 0.04485]  # Class presence
)
# The higher the presence, the lower the weight
base_weight = 0.25
class_loss_weight = torch.Tensor(class_rate.max() / class_rate * base_weight)
print(f'Using class weights: {class_loss_weight}')

# ---------------
# Model
# ---------------

if load_from is not None:
    filename = os.path.basename(load_from)
    iteration_offset = int(os.path.splitext(filename)[0].split('_')[-1])
    print(f'Loading weights from checkpoint: {load_from}. '
          f'Iteration: {iteration_offset}')
    unet = torch.load(load_from)
else:
    print('Initializing model from scratch')
    unet = Unet(n_channels=n_channels, n_classes=n_classes)
    unet.apply(initialize_weights)
    iteration_offset = 0


optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=0.99)
cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_loss_weight)

if cuda:
    cross_entropy_loss.cuda()
    unet.cuda()


# ---------------
# Training dataset
# ---------------

data_aug = DataAugmentation(
    central_crop_size=image_crop,
    h_flipping_chance=0.50,
    brightness_rate=0.10,
    contrast_rate=0.10,
    saturation_rate=0.10,
    hue_rate=0.05)

train_dataset = DeepFashion2Dataset(DATA_PATH,
                                    subset='train',
                                    image_resize=image_resize,
                                    data_augmentation=data_aug)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=data_load_workers,
                          pin_memory=True)

batches_per_epoch = len(train_loader)

# ---------------
# Validation dataset
# ---------------

val_dataset = DeepFashion2Dataset(DATA_PATH,
                                  subset='train',
                                  image_resize=image_crop)

val_loader = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=data_load_workers,
                        pin_memory=True)


# ---------------
# Training loop
# ---------------


train_writer = SummaryWriter(os.path.join(tensorboard_dir, 'training'),
                             filename_suffix='_train')
val_writer = SummaryWriter(os.path.join(tensorboard_dir, 'validation'),
                           filename_suffix='_validation')

for epoch in range(n_epochs):

    for i, (imgs, masks) in enumerate(train_loader):

        iteration = epoch * batches_per_epoch + i + iteration_offset

        # Configure batch data
        batch_imgs = imgs.to(device).float()
        batch_masks = masks.to(device).long()

        # Compute loss and propagate backwards
        predicted_masks = unet(batch_imgs)
        loss = cross_entropy_loss(predicted_masks, batch_masks)

        # Set cumulative gradients in optimizer to 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        iou_val = iou(batch_masks, predicted_masks)

        if iteration % stats_interval == 0:
            training_progress = Evaluation(
                iteration=iteration,
                images=imgs,
                masks=torch.unsqueeze(masks, dim=1),
                predictions=torch.unsqueeze(
                    torch.argmax(predicted_masks, dim=1), dim=0),
                loss_value=loss.item(),
                iou_value=iou_val,
            )
            training_progress.track(train_writer)
            validation_progress = _validation_eval(unet,
                                                   cross_entropy_loss,
                                                   val_loader,
                                                   iteration)
            validation_progress.track(val_writer)

        if iteration % save_model_interval == 0:
            torch.save(unet, f'unet_iter_{iteration}.pt')

        print(
            "[Epoch %d/%d] [Batch %d/%d] Loss: %f, Iou: %f"
            % (epoch, n_epochs, i, batches_per_epoch, loss.item(), iou_val)
        )


torch.save(unet, 'unet.pt')
