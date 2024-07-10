from itertools import repeat
from pathlib import Path

from modules.dataset import DATASET_DIR, DATASET_ORIG_DIR, UNIQUE_PLANTS
from modules.preprocess import create_rcnn_dataset

if __name__ == "__main__":
    DATASET_RAW: list[tuple[int, Path]] = []

    # with open(DATASET_DIR / "rcnn_dataset.csv", "w", encoding="utf-8") as f:
    # writer = csv.writer(f)
    for plant_class in DATASET_ORIG_DIR.iterdir():
        print(plant_class.name)
        index = UNIQUE_PLANTS.index(plant_class.name)
        # for c, plant in zip(repeat(index), plant_class.iterdir()):
        #     data = get_image_data(c, plant)
        #     writer.writerow(data)
        DATASET_RAW.extend(list(zip(repeat(index), plant_class.iterdir())))

    DATASET = create_rcnn_dataset(DATASET_RAW)
    DATASET.to_csv(DATASET_DIR / "rcnn_dataset_2.csv")

    # DATASET = create_rcnn_dataset_bin(DATASET_RAW)
    # print("\n\n\n", DATASET, "\n\n\n")
    # with open(DATASET_DIR / "rcnn_dataset.bin", "wb") as f:
    #     f.write(DATASET.tobytes())
