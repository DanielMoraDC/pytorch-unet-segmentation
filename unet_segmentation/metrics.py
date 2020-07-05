import torch
import numpy as np


def mean_iou(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """ Computes the mean of the Intersection over union of all classes """
    return np.mean(iou(y_true, y_pred))


def iou(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """ Computes the Intersection over all classes of the inputs """
    true_classes = y_true.unique().detach()
    predicted_image = torch.argmax(y_pred, dim=1)

    def _class_iou(c):
        predicted_class_mask = torch.eq(predicted_image, c).bool()
        true_class_mask = torch.eq(y_true, c).bool()
        intersection = predicted_class_mask & true_class_mask
        union = predicted_class_mask | true_class_mask
        return (intersection.float().sum() / union.float().sum()).mean()

    # Iterate over true classes so we never have a zero division
    return [_class_iou(c).item() for c in true_classes if c != 0]
