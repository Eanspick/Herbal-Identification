"""
This type stub file was generated by pyright.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple, Union
from .vision import VisionDataset

class StanfordCars(VisionDataset):
    """Stanford Cars  Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    The original URL is https://ai.stanford.edu/~jkrause/cars/car_dataset.html, but it is broken.

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): This parameter exists for backward compatibility but it does not
            download the dataset, since the original URL is not available anymore. The dataset
            seems to be available on Kaggle so you can try to manually download it using
            `these instructions <https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616>`_.
    """
    def __init__(self, root: Union[str, pathlib.Path], split: str = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        ...
    
    def download(self):
        ...
    

