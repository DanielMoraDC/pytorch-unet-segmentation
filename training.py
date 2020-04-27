from unet import (
    Unet,
    initialize_weights
)

from data import (
    DeepFashion2Dataset,
    DataAugmentation
)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DATA_PATH = 'seg_dataset'

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False


# ---------------
# Parameters
# ---------------

n_epochs = 100
batch_size = 1
n_channels = 3
n_classes = 6 + 1  # background
data_load_workers = 2
lr = 0.001
tensorboard_dir = 'logs'
stats_interval = 10


# ---------------
# Model
# ---------------

cross_entropy_loss = torch.nn.CrossEntropyLoss()
unet = Unet(n_channels=n_channels, n_classes=n_classes)
optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=0.99)

if cuda:
    cross_entropy_loss.cuda()
    unet.cuda()

unet.apply(initialize_weights)


# ---------------
# Dataset
# ---------------

data_aug = DataAugmentation(
    central_crop_size=512,
    h_flipping_chance=0.50,
    brightness_rate=0.10,
    contrast_rate=0.10,
    saturation_rate=0.10,
    hue_rate=0.05)

dataset = DeepFashion2Dataset(DATA_PATH,
                              image_resize=575,
                              data_augmentation=data_aug)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=data_load_workers)

batches_per_epoch = len(dataloader)


# ---------------
# Training loop
# ---------------


writer = SummaryWriter(tensorboard_dir)
for epoch in range(n_epochs):

    for i, (imgs, masks) in enumerate(dataloader):

        iteration = epoch * batches_per_epoch + i

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

        if iteration % stats_interval == 0:
            writer.add_scalar('Loss', loss.item(), iteration)
            writer.add_images('Inputs', batch_imgs, iteration)
            predicted_images = torch.argmax(predicted_masks,
                                            dim=1).unsqueeze_(1)
            writer.add_images('Predicted', predicted_images, iteration)
            writer.add_images('Groundtruth', masks.unsqueeze_(1), iteration)

        print(
            "[Epoch %d/%d] [Batch %d/%d] Loss: %f"
            % (epoch, n_epochs, i, batches_per_epoch, loss.item())
        )


torch.save(unet, 'unet.pt')
