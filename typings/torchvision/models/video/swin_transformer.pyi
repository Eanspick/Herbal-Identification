"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Callable, List, Optional
from torch import Tensor, nn
from .._api import WeightsEnum, register_model
from .._utils import handle_legacy_interface

__all__ = ["SwinTransformer3d", "Swin3D_T_Weights", "Swin3D_S_Weights", "Swin3D_B_Weights", "swin3d_t", "swin3d_s", "swin3d_b"]
def shifted_window_attention_3d(input: Tensor, qkv_weight: Tensor, proj_weight: Tensor, relative_position_bias: Tensor, window_size: List[int], num_heads: int, shift_size: List[int], attention_dropout: float = ..., dropout: float = ..., qkv_bias: Optional[Tensor] = ..., proj_bias: Optional[Tensor] = ..., training: bool = ...) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, T, H, W, C]): The input tensor, 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): 3-dimensions window size, T, H, W .
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention (T, H, W).
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[B, T, H, W, C]: The output tensor after shifted window attention.
    """
    ...

class ShiftedWindowAttention3d(nn.Module):
    """
    See :func:`shifted_window_attention_3d`.
    """
    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool = ..., proj_bias: bool = ..., attention_dropout: float = ..., dropout: float = ...) -> None:
        ...
    
    def define_relative_position_bias_table(self) -> None:
        ...
    
    def define_relative_position_index(self) -> None:
        ...
    
    def get_relative_position_bias(self, window_size: List[int]) -> torch.Tensor:
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        ...
    


class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size: List[int], in_channels: int = ..., embed_dim: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None:
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        ...
    


class SwinTransformer3d(nn.Module):
    """
    Implements 3D Swin Transformer from the `"Video Swin Transformer" <https://arxiv.org/abs/2106.13230>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 400.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
        patch_embed (nn.Module, optional): Patch Embedding layer. Default: None.
    """
    def __init__(self, patch_size: List[int], embed_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], mlp_ratio: float = ..., dropout: float = ..., attention_dropout: float = ..., stochastic_depth_prob: float = ..., num_classes: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., block: Optional[Callable[..., nn.Module]] = ..., downsample_layer: Callable[..., nn.Module] = ..., patch_embed: Optional[Callable[..., nn.Module]] = ...) -> None:
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        ...
    


_COMMON_META = ...
class Swin3D_T_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    DEFAULT = ...


class Swin3D_S_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    DEFAULT = ...


class Swin3D_B_Weights(WeightsEnum):
    KINETICS400_V1 = ...
    KINETICS400_IMAGENET22K_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin3D_T_Weights.KINETICS400_V1))
def swin3d_t(*, weights: Optional[Swin3D_T_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_tiny architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_T_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", Swin3D_S_Weights.KINETICS400_V1))
def swin3d_s(*, weights: Optional[Swin3D_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_small architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_S_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", Swin3D_B_Weights.KINETICS400_V1))
def swin3d_b(*, weights: Optional[Swin3D_B_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_base architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_B_Weights
        :members:
    """
    ...
