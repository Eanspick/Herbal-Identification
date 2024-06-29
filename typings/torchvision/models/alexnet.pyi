"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from typing import Any, Optional
from ._api import WeightsEnum, register_model
from ._utils import handle_legacy_interface

__all__ = ["AlexNet", "AlexNet_Weights", "alexnet"]
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = ..., dropout: float = ...) -> None:
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    


class AlexNet_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", AlexNet_Weights.IMAGENET1K_V1))
def alexnet(*, weights: Optional[AlexNet_Weights] = ..., progress: bool = ..., **kwargs: Any) -> AlexNet:
    """AlexNet model architecture from `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`__.

    .. note::
        AlexNet was originally introduced in the `ImageNet Classification with
        Deep Convolutional Neural Networks
        <https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`__
        paper. Our implementation is based instead on the "One weird trick"
        paper above.

    Args:
        weights (:class:`~torchvision.models.AlexNet_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.AlexNet_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.AlexNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.AlexNet_Weights
        :members:
    """
    ...
