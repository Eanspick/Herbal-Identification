"""
This type stub file was generated by pyright.
"""

import torch
import torch.fx
from torch import Tensor, nn

@torch.fx.wrap
def ps_roi_align(input: Tensor, boxes: Tensor, output_size: int, spatial_scale: float = ..., sampling_ratio: int = ...) -> Tensor:
    """
    Performs Position-Sensitive Region of Interest (RoI) Align operator
    mentioned in Light-Head R-CNN.

    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the box coordinates to
            the input coordinates. For example, if your boxes are defined on the scale
            of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
            the original image), you'll want to set this to 0.5. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1

    Returns:
        Tensor[K, C / (output_size[0] * output_size[1]), output_size[0], output_size[1]]: The pooled RoIs
    """
    ...

class PSRoIAlign(nn.Module):
    """
    See :func:`ps_roi_align`.
    """
    def __init__(self, output_size: int, spatial_scale: float, sampling_ratio: int) -> None:
        ...
    
    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        ...
    
    def __repr__(self) -> str:
        ...
    

