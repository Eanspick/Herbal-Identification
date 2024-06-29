"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
from .vision import VisionDataset

class SBDataset(VisionDataset):
    """`Semantic Boundaries Dataset <http://home.bharathh.info/pubs/codes/SBD/download.html>`_

    The SBD currently contains annotations from 11355 images taken from the PASCAL VOC 2011 dataset.

    .. note ::

        Please note that the train and val splits included with this dataset are different from
        the splits in the PASCAL VOC dataset. In particular some "train" images might be part of
        VOC2012 val.
        If you are interested in testing on VOC 2012 val, then use `image_set='train_noval'`,
        which excludes all val images.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Semantic Boundaries Dataset
        image_set (string, optional): Select the image_set to use, ``train``, ``val`` or ``train_noval``.
            Image set ``train_noval`` excludes VOC 2012 val images.
        mode (string, optional): Select target type. Possible values 'boundaries' or 'segmentation'.
            In case of 'boundaries', the target is an array of shape `[num_classes, H, W]`,
            where `num_classes=20`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version. Input sample is PIL image and target is a numpy array
            if `mode='boundaries'` or PIL image if `mode='segmentation'`.
    """
    url = ...
    md5 = ...
    filename = ...
    voc_train_url = ...
    voc_split_filename = ...
    voc_split_md5 = ...
    def __init__(self, root: Union[str, Path], image_set: str = ..., mode: str = ..., download: bool = ..., transforms: Optional[Callable] = ...) -> None:
        ...
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        ...
    
    def __len__(self) -> int:
        ...
    
    def extra_repr(self) -> str:
        ...
    


