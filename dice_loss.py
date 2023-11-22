import torch
from fastai.torch_core import TensorBase
from torch import Tensor


class DiceLoss:
    "Dice loss for segmentation"

    def __init__(self,
                 axis: int = 1,  # Class axis
                 smooth: float = 1e-6,  # Helps with numerical stabilities in the IoU division
                 reduction: str = "sum",  # PyTorch reduction to apply to the output
                 square_in_union: bool = False  # Squares predictions to increase slope of gradients
                 ):
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union

    def __call__(self, pred: Tensor, targ: Tensor) -> Tensor:
        "One-hot encodes targ, then runs IoU calculation then takes 1-dice value"
        targ = self._one_hot(targ, pred.shape[self.axis])
        pred, targ = TensorBase(pred), TensorBase(targ)
        assert pred.shape == targ.shape, 'input and target dimensions differ, DiceLoss expects non one-hot targs'
        pred = self.activation(pred)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = (torch.sum(pred ** 2 + targ, dim=sum_dims) if self.square_in_union
                 else torch.sum(pred + targ, dim=sum_dims))
        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def _one_hot(
            x: Tensor,  # Non one-hot encoded targs
            classes: int,  # The number of classes
            axis: int = 1  # The axis to stack for encoding (class dimension)
    ) -> Tensor:
        "Creates one binary mask per class"
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], dim=axis)

    def activation(self, x: Tensor) -> Tensor:
        "Activation function applied to model output"
        return torch.softmax(x, dim=self.axis)

    def decodes(self, x: Tensor) -> Tensor:
        "Converts model output to target format"
        return x.argmax(dim=self.axis)
