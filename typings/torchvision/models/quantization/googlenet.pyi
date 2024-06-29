"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional, Union
from torch import Tensor
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface
from ..googlenet import BasicConv2d, GoogLeNet, GoogLeNetOutputs, GoogLeNet_Weights, Inception, InceptionAux

__all__ = ["QuantizableGoogLeNet", "GoogLeNet_QuantizedWeights", "googlenet"]
class QuantizableBasicConv2d(BasicConv2d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        ...
    
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None:
        ...
    


class QuantizableInception(Inception):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        ...
    


class QuantizableInceptionAux(InceptionAux):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        ...
    


class QuantizableGoogLeNet(GoogLeNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        ...
    
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None:
        r"""Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        ...
    


class GoogLeNet_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = ...
    DEFAULT = ...


@register_model(name="quantized_googlenet")
@handle_legacy_interface(weights=("pretrained", lambda kwargs: GoogLeNet_QuantizedWeights.IMAGENET1K_FBGEMM_V1 if kwargs.get("quantize", False) else GoogLeNet_Weights.IMAGENET1K_V1))
def googlenet(*, weights: Optional[Union[GoogLeNet_QuantizedWeights, GoogLeNet_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableGoogLeNet:
    """GoogLeNet (Inception v1) model architecture from `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`__.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.GoogLeNet_QuantizedWeights` or :class:`~torchvision.models.GoogLeNet_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.GoogLeNet_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool, optional): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableGoogLeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/googlenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.GoogLeNet_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.GoogLeNet_Weights
        :members:
        :noindex:
    """
    ...

