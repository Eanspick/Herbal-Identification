from itertools import repeat
from pathlib import Path

from modules.dataset import DATASET_DIR, DATASET_PROCESSED_DIR, UNIQUE_PLANTS_PROCESSED
from modules.preprocess import create_dataset_2

if __name__ == "__main__":
    DATASET_RAW: list[tuple[int, Path]] = []

    for plant_class in DATASET_PROCESSED_DIR.glob("*"):
        print(plant_class.name)
        index = UNIQUE_PLANTS_PROCESSED.index(plant_class.name)
        DATASET_RAW.extend(list(zip(repeat(index), plant_class.glob("*"))))

    DATASET = create_dataset_2(DATASET_RAW)
    DATASET.to_csv(DATASET_DIR / "processed_features.csv")
