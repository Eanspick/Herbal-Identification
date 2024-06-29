"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, Dict, List, Optional
from torch import Tensor, nn
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface
from ..mobilenetv3 import MobileNet_V3_Large_Weights
from .ssd import SSD, SSDScoringHead

__all__ = ["SSDLite320_MobileNet_V3_Large_Weights", "ssdlite320_mobilenet_v3_large"]
class SSDLiteHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: Callable[..., nn.Module]) -> None:
        ...
    
    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        ...
    


class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: Callable[..., nn.Module]) -> None:
        ...
    


class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], norm_layer: Callable[..., nn.Module]) -> None:
        ...
    


class SSDLiteFeatureExtractorMobileNet(nn.Module):
    def __init__(self, backbone: nn.Module, c4_pos: int, norm_layer: Callable[..., nn.Module], width_mult: float = ..., min_depth: int = ...) -> None:
        ...
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        ...
    


class SSDLite320_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", SSDLite320_MobileNet_V3_Large_Weights.COCO_V1), weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1))
def ssdlite320_mobilenet_v3_large(*, weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[MobileNet_V3_Large_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., **kwargs: Any) -> SSD:
    """SSDlite model architecture with input size 320x320 and a MobileNetV3 Large backbone, as
    described at `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__ and
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`__.

    .. betastatus:: detection module

    See :func:`~torchvision.models.detection.ssd300_vgg16` for more details.

    Example:

        >>> model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model
            (including the background).
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers
            starting from final block. Valid values are between 0 and 6, with 6 meaning all
            backbone layers are trainable. If ``None`` is passed (the default) this value is
            set to 6.
        norm_layer (callable, optional): Module specifying the normalization layer to use.
        **kwargs: parameters passed to the ``torchvision.models.detection.ssd.SSD``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssdlite.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights
        :members:
    """
    ...
