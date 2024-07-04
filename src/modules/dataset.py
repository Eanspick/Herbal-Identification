from pathlib import Path

import torchvision.transforms as transforms
from torch import Tensor

BASEDIR = Path(__file__).parent.parent.parent.resolve()
LOG_DIR = BASEDIR / "logs"
DATASET_BASEDIR = BASEDIR / "dataset"

DATASET_DIR = DATASET_BASEDIR / "New Plant-Dataset"
DATASET_ORIG_DIR = DATASET_DIR / "original"
DATASET_AUGMENTED_DIR = DATASET_DIR / "augmented"
DATASET_PROCESSED_DIR = DATASET_DIR / "processed"

UNIQUE_PLANTS: list[str] = []
for i in DATASET_AUGMENTED_DIR.glob("*"):
    UNIQUE_PLANTS.append(i.name)


FLAVIA_DIR = DATASET_BASEDIR / "Flavia"

UNIQUE_FLAVIA_PLANTS: list[str] = []
for i in FLAVIA_DIR.glob("*"):
    UNIQUE_FLAVIA_PLANTS.append(i.name)


RESIZE = transforms.Resize((256, 256))
NORMALIZE = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Define transformation
TRANSFORM: "transforms.Compose[Tensor]" = transforms.Compose(
    [
        # SubtractBackground(),
        RESIZE,
        transforms.ToTensor(),
        NORMALIZE,
    ]
)

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def inv_norm(tensor: Tensor) -> Tensor:
    return INV_NORMALIZE(tensor)
