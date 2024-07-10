from typing import Generic, overload

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing_extensions import TypeVar

T = TypeVar("T", bound=torch.Tensor | nn.Module)
T_co = TypeVar("T_co", covariant=True, default=torch.Tensor)


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


class DeviceDataLoader(Generic[T_co]):
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl: DataLoader[T_co], device: str | torch.device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        b: list[torch.Tensor]
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
