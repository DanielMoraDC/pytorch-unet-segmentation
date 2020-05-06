from unet import (
    Unet,
    initialize_weights
)

from metrics import iou

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

load_from = 'unet_iter_300000.pt'
n_epochs = 250
batch_size = 1
n_channels = 3
n_classes = 6 + 1  # background
data_load_workers = 2
lr = 0.001
tensorboard_dir = 'logs'
stats_interval = 2500
save_model_interval = 50000
image_crop = 512
image_resize = 575

# Create weights for weighted loss
base_weight = 0.25
class_rate = np.array(
    [0.50] +  # Set high value to background
    [0.4039, 0.1173, 0.1587, 0.0988, 0.1774, 0.04485]  # Class presence
)
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
# Dataset
# ---------------

data_aug = DataAugmentation(
    central_crop_size=image_crop,
    h_flipping_chance=0.50,
    brightness_rate=0.10,
    contrast_rate=0.10,
    saturation_rate=0.10,
    hue_rate=0.05)

dataset = DeepFashion2Dataset(DATA_PATH,
                              image_resize=image_resize,
                              data_augmentation=data_aug)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=data_load_workers,
                        pin_memory=True)

batches_per_epoch = len(dataloader)


# ---------------
# Training loop
# ---------------


writer = SummaryWriter(tensorboard_dir)
for epoch in range(n_epochs):

    for i, (imgs, masks) in enumerate(dataloader):

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
            writer.add_scalar('Loss', loss.item(), iteration)
            writer.add_scalar('Iou', iou_val, iteration)
            writer.add_images('Inputs', batch_imgs, iteration)
            predicted_images = torch.argmax(predicted_masks,
                                            dim=1).unsqueeze_(1)
            writer.add_images('Predicted', predicted_images, iteration)
            writer.add_images('Groundtruth', masks.unsqueeze_(1), iteration)

        if iteration % save_model_interval == 0:
            torch.save(unet, f'unet_iter_{iteration}.pt')

        print(
            "[Epoch %d/%d] [Batch %d/%d] Loss: %f, Iou: %f"
            % (epoch, n_epochs, i, batches_per_epoch, loss.item(), iou_val)
        )


torch.save(unet, 'unet.pt')
