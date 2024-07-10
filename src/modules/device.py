from typing import Generic, cast, overload

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing_extensions import TypeVar, TypeVarTuple

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
T_co = TypeVar("T_co", covariant=True, default=torch.Tensor)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_default_device()


@overload
def to_device(tensor: tuple[*Ts], device: str | torch.device) -> tuple[*Ts]: ...
@overload
def to_device(tensor: list[T], device: str | torch.device) -> list[T]: ...
@overload
def to_device(tensor: T, device: str | torch.device) -> T: ...


def to_device(
    tensor: T | list[T] | tuple[*Ts], device: str | torch.device
) -> T | list[T] | tuple[*Ts]:
    """Move tensor(s) to chosen device"""
    if isinstance(tensor, (torch.Tensor, nn.Module)):
        return tensor.to(device, non_blocking=True)
    if isinstance(tensor, tuple):
        tensor = cast(tuple[*Ts], tensor)
        return tuple(to_device(x, device) for x in tensor)  # type: ignore
    if isinstance(tensor, list):
        tensor = cast(list[T], tensor)
        return [to_device(x, device) for x in tensor]
    return tensor


class DeviceDataLoader(Generic[T_co]):
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl: DataLoader[T_co], device: str | torch.device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        b: list[T_co]
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
