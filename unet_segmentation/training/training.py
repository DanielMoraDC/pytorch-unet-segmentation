import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet_segmentation.unet import (
    Unet,
    initialize_weights
)

from unet_segmentation.metrics import mean_iou

from unet_segmentation.training.stats import (
    Stats,
    SummaryStats,
    MovingStats
)


@dataclass
class TrainingParams(object):

    train_loader: DataLoader
    val_loader: DataLoader
    loss: torch.nn.Module
    lr: float
    momentum: float
    stats_interval: int
    save_model_interval: int
    save_model_dir: str
    n_epochs: int
    n_classes: int
    device: torch.device
    n_channels: int = 3
    tensorboard_logs_dir: str = 'logs'
    checkpoint: str = None


def _validation_eval(model: Unet,
                     params: TrainingParams,
                     n_samples: int = 5) -> Stats:
    """ Extract stats from a forward pass for whole validation dataset
    Args:
        model: Model used to extract masks from data
        params: Training parameters
        n_samples: Number of image samples to extract in order to include them
            in the generated report.
    Returns:
        stats: Validation stats report.
        """
    losses, ious = [], []
    image_samples, mask_samples, prediction_samples = [], [], []
    sample_idxs = np.random.randint(0, len(params.val_loader), size=n_samples)

    with torch.no_grad():

        for i, (imgs, masks) in enumerate(params.val_loader):

            batch_imgs = imgs.to(params.device).float()
            batch_masks = masks.to(params.device).long()

            predicted_masks = model(batch_imgs)
            loss = params.loss(predicted_masks, batch_masks)

            losses.append(loss)
            ious.append(mean_iou(batch_masks, predicted_masks))

            if i in sample_idxs:
                image_samples.append(imgs)
                mask_samples.append(masks)
                prediction_samples.append(predicted_masks)

    return Stats(
        images=torch.cat(image_samples, dim=0),
        masks=torch.unsqueeze(torch.cat(mask_samples, dim=0), dim=1),
        predictions=torch.cat(prediction_samples, dim=0),
        loss=torch.mean(torch.Tensor(losses)),
        iou_value=np.mean(ious)
    )


def _train_step(image_batch: torch.Tensor,
                mask_batch: torch.Tensor,
                model: Unet,
                optimizer: torch.optim.Optimizer,
                params: TrainingParams) -> Stats:

    # Compute loss and propagate backwards
    predicted_masks = model(image_batch)
    loss = params.loss(predicted_masks, mask_batch)

    # Set cumulative gradients in optimizer to 0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return Stats(
        images=image_batch,
        masks=torch.unsqueeze(mask_batch, dim=1),
        predictions=predicted_masks,
        loss=loss,
        iou_value=mean_iou(mask_batch, predicted_masks)
    )


def _print_current_step(epoch: int,
                        batch_idx: int,
                        train_stats: Stats,
                        params: TrainingParams) -> None:
    n_batches = len(params.train_loader)
    print(
        f'[Epoch {epoch}/{params.n_epochs}] [Batch {batch_idx}/{n_batches}] ' +
        f'Loss: {train_stats.loss.item():.4f}, Iou: {train_stats.iou_value:.2f}'  # noqa
    )


def _print_stats(epoch: int,
                 stats: Stats,
                 params: TrainingParams,
                 tag: str = 'training') -> None:
    print(
        f'[Epoch {epoch}/{params.n_epochs}] ' +
        f'{tag} loss: {stats.loss.item():.4f}, ' +
        f'{tag} Iou: {stats.iou_value:.2f}'
    )


def _load_model(params: TrainingParams) -> Tuple[Unet, int]:
    checkpoint = params.checkpoint
    if checkpoint is not None:
        filename = os.path.basename(checkpoint)
        iteration_offset = int(os.path.splitext(filename)[0].split('_')[-1])
        print(
            f'Loading weights from checkpoint: {checkpoint}. '
              f'Iteration: {iteration_offset}'
        )
        unet = torch.load(checkpoint)
    else:
        print('Initializing model from scratch')
        unet = Unet(n_channels=params.n_channels, n_classes=params.n_classes)
        unet.apply(initialize_weights)
        iteration_offset = 0

    return unet, iteration_offset


def _initialize_writers(params: TrainingParams) -> Tuple[SummaryWriter, SummaryWriter]:  # noqa
    def _init_writer(tag: str) -> SummaryWriter:
        return SummaryWriter(
            os.path.join(params.tensorboard_logs_dir, tag),
            filename_suffix=f'_{tag}'
        )

    return _init_writer('train'), _init_writer('validation')


def fit(params: TrainingParams) -> None:

    unet, iteration_offset = _load_model(params)
    optimizer = torch.optim.SGD(
        unet.parameters(), lr=params.lr, momentum=params.momentum)

    # Move model and loss to devices
    params.loss.to(params.device)
    unet.to(params.device)

    n_batches = len(params.train_loader)
    train_writer, val_writer = _initialize_writers(params)
    stats = MovingStats()

    for epoch in range(params.n_epochs):

        for batch_idx, (imgs, masks) in enumerate(params.train_loader):

            # Configure batch data
            iteration = epoch * n_batches + batch_idx + iteration_offset
            batch_imgs = imgs.to(params.device).float()
            batch_masks = masks.to(params.device).long()

            training_stats = _train_step(image_batch=batch_imgs,
                                         mask_batch=batch_masks,
                                         model=unet,
                                         optimizer=optimizer,
                                         params=params)
            stats.update(training_stats.loss, training_stats.iou_value)

            if iteration % params.stats_interval == 0 \
                    and iteration != iteration_offset:
                # Replace step metrics with moving metrics
                training_stats.loss = stats.moving_loss()
                training_stats.iou_value = stats.moving_iou()
                stats.restart()

                # Track training summary
                training_summary = SummaryStats.from_stats(training_stats,
                                                           iteration)
                training_summary.track(train_writer)

                # Track validation summary
                validation_stats = _validation_eval(model=unet, params=params)
                validation_summary = SummaryStats.from_stats(validation_stats,
                                                             iteration)
                validation_summary.track(val_writer)

                _print_stats(epoch, training_stats, params)
                _print_stats(epoch, validation_stats, params, tag='validation')

            if iteration % params.save_model_interval == 0 and \
                    iteration != iteration_offset:
                model_path = os.path.join(params.save_model_dir,
                                          f'unet_iter_{iteration}.pt')
                torch.save(unet, model_path)

    torch.save(unet, os.path.join(params.save_model_dir, f'unet_final.pt'))
