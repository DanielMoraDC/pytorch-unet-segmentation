import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class MovingStats(object):

    losses: List
    ious: List

    def __init__(self):
        self.restart()

    def update(self, loss: float, iou: float) -> None:
        self.losses.append(loss)
        self.ious.append(iou)

    def moving_loss(self):
        return torch.mean(torch.Tensor(self.losses))

    def moving_iou(self):
        return np.mean(self.ious)

    def restart(self):
        self.losses = []
        self.ious = []


@dataclass
class Stats(object):

    images: torch.Tensor
    masks: torch.Tensor
    predictions: torch.Tensor

    loss: torch.Tensor
    iou_value: float


@dataclass
class SummaryStats(object):

    iteration: int

    images: np.ndarray
    masks: np.ndarray
    predictions: np.ndarray

    loss_value: float
    iou_value: float

    def track(self, writer: SummaryWriter) -> None:
        writer.add_images('inputs', self.images, self.iteration)
        writer.add_images('groundtruth', self.masks, self.iteration)
        writer.add_images('predictions', self.predictions, self.iteration)
        writer.add_scalar('loss', self.loss_value, self.iteration)
        writer.add_scalar('iou', self.iou_value, self.iteration)

    @staticmethod
    def from_stats(stats: Stats, iteration: int):
        return SummaryStats(
            iteration=iteration,
            images=stats.images,
            masks=stats.masks,
            predictions=torch.unsqueeze(
                torch.argmax(stats.predictions, dim=1), dim=1),
            loss_value=stats.loss.item(),
            iou_value=stats.iou_value,
        )
