from typing import overload

import torch
import torch.nn as nn
from typing_extensions import TypeVar

T = TypeVar("T", bound=torch.Tensor | nn.Module)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_default_device()


@overload
def to_device(
    tensor: list[T] | tuple[T, ...], device: str | torch.device
) -> list[T]: ...
@overload
def to_device(tensor: T, device: str | torch.device) -> T: ...


def to_device(
    tensor: T | list[T] | tuple[T, ...], device: str | torch.device
) -> list[T] | T:
    """Move tensor(s) to chosen device"""
    if isinstance(tensor, (list, tuple)):
        return [x.to(device, non_blocking=True) for x in tensor]
    return tensor.to(device, non_blocking=True)
