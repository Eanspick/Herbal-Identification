from pathlib import Path

BASEDIR = Path(__file__).parent.parent.resolve()
DATASET_BASEDIR = BASEDIR.parent / "dataset"

DATASET_DIR = DATASET_BASEDIR / "New Plant-Dataset"
DATASET_ORIG_DIR = DATASET_DIR / "original"
DATASET_AUGMENTED_DIR = DATASET_DIR / "augmented"

UNIQUE_PLANTS: list[str] = []
for i in DATASET_AUGMENTED_DIR.glob("*"):
    UNIQUE_PLANTS.append(i.name)


FLAVIA_DIR = DATASET_BASEDIR / "Flavia"

UNIQUE_FLAVIA_PLANTS: list[str] = []
for i in FLAVIA_DIR.glob("*"):
    UNIQUE_FLAVIA_PLANTS.append(i.name)
