"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional
from torch import nn
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface
from ..resnet import ResNet101_Weights, ResNet50_Weights
from ._utils import _SimpleSegmentationModel

__all__ = ["FCN", "FCN_ResNet50_Weights", "FCN_ResNet101_Weights", "fcn_resnet50", "fcn_resnet101"]
class FCN(_SimpleSegmentationModel):
    """
    Implements FCN model from
    `"Fully Convolutional Networks for Semantic Segmentation"
    <https://arxiv.org/abs/1411.4038>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    ...


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        ...
    


_COMMON_META = ...
class FCN_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = ...
    DEFAULT = ...


class FCN_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1), weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1))
def fcn_resnet50(*, weights: Optional[FCN_ResNet50_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., aux_loss: Optional[bool] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., **kwargs: Any) -> FCN:
    """Fully-Convolutional Network model with a ResNet-50 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet50_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1), weights_backbone=("pretrained_backbone", ResNet101_Weights.IMAGENET1K_V1))
def fcn_resnet101(*, weights: Optional[FCN_ResNet101_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., aux_loss: Optional[bool] = ..., weights_backbone: Optional[ResNet101_Weights] = ..., **kwargs: Any) -> FCN:
    """Fully-Convolutional Network model with a ResNet-101 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet101_Weights
        :members:
    """
    ...
