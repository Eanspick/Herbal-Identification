"""
This type stub file was generated by pyright.
"""

import os
import warnings
import torch
from modulefinder import Module
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
from .extension import _HAS_OPS
from .version import __version__

if not _HAS_OPS and os.path.dirname(os.path.realpath(__file__)) == os.path.join(os.path.realpath(os.getcwd()), "torchvision"):
    message = ...
_image_backend = ...
_video_backend = ...
def set_image_backend(backend): # -> None:
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'PIL', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    ...

def get_image_backend(): # -> str:
    """
    Gets the name of the package used to load images
    """
    ...

def set_video_backend(backend): # -> None:
    """
    Specifies the package used to decode videos.

    Args:
        backend (string): Name of the video backend. one of {'pyav', 'video_reader'}.
            The :mod:`pyav` package uses the 3rd party PyAv library. It is a Pythonic
            binding for the FFmpeg libraries.
            The :mod:`video_reader` package includes a native C++ implementation on
            top of FFMPEG libraries, and a python API of TorchScript custom operator.
            It generally decodes faster than :mod:`pyav`, but is perhaps less robust.

    .. note::
        Building with FFMPEG is disabled by default in the latest `main`. If you want to use the 'video_reader'
        backend, please compile torchvision from source.
    """
    ...

def get_video_backend(): # -> str:
    """
    Returns the currently active video backend used to decode videos.

    Returns:
        str: Name of the video backend. one of {'pyav', 'video_reader'}.
    """
    ...

def disable_beta_transforms_warning(): # -> None:
    ...
