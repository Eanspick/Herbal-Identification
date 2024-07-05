import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path

import cv2
import mahotas as mt  # type: ignore
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from PIL.Image import Image, fromarray

from .dataset import DATASET_BASEDIR, DATASET_PROCESSED_DIR

KERNEL = np.ones((50, 50), np.uint8)

NAMES = [
    "class",
    "area",
    "perimeter",
    "physiological_length",
    "physiological_width",
    "aspect_ratio",
    "rectangularity",
    "circularity",
    "mean_r",
    "mean_g",
    "mean_b",
    "stddev_r",
    "stddev_g",
    "stddev_b",
    "contrast",
    "correlation",
    "inverse_difference_moments",
    "entropy",
]


def find_contour(image: MatLike):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(grayscale, (55, 55), 0)
    _, thresholded_image = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    closing = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = list(contours)
    contours.sort(key=cv2.contourArea, reverse=True)

    y_ri, x_ri, _ = image.shape
    contains = [
        cv2.pointPolygonTest(c, (x_ri // 2, y_ri // 2), False) for c in contours
    ]
    val = [contains.index(i) for i in contains if i > 0]
    return contours[val[0]] if val else contours[0]


def extract_features(class_: int, image: MatLike):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Shape features
    contour = find_contour(image)
    # M = cv2.moments(cnt)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    rectangularity = w * h / area
    circularity = ((perimeter) ** 2) / area

    # Color features
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    # Texture features
    textures = mt.features.haralick(gs)  # type: ignore
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

    vector = (
        class_,
        area,
        perimeter,
        w,
        h,
        aspect_ratio,
        rectangularity,
        circularity,
        red_mean,
        green_mean,
        blue_mean,
        red_std,
        green_std,
        blue_std,
        contrast,
        correlation,
        inverse_diff_moments,
        entropy,
    )
    return vector


def subtract_background(image: MatLike):
    height, width, _ = image.shape
    image = cv2.resize(image, (2000, 2000))
    black_img = np.empty((2000, 2000, 3), dtype=np.uint8)

    contour = find_contour(image)
    mask = cv2.drawContours(black_img, [contour], 0, (255, 255, 255), -1)
    masked_img = cv2.bitwise_and(image, mask)
    final_img = np.where(masked_img.any(-1, keepdims=True), masked_img, 255)
    return cv2.resize(final_img, (width, height))


def extract_features_file(class_: int, file: Path):
    print(f"Preprocessing file {file.relative_to(DATASET_BASEDIR)}")
    image = cv2.imread(str(file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return extract_features(class_, subtract_background(image))


def create_dataset(images: Sequence[tuple[int, Path]]):
    with mp.Pool() as pool:
        results = pool.starmap(extract_features_file, images, chunksize=1)
        return pd.DataFrame.from_records(results, columns=NAMES)  # type: ignore


def extract_features_file_2(class_: int, file: Path):
    print(f"Preprocessing file {file.relative_to(DATASET_BASEDIR)}")
    image = cv2.imread(str(file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return extract_features(class_, image)


def create_dataset_2(images: Sequence[tuple[int, Path]]):
    with mp.Pool() as pool:
        results = pool.starmap(extract_features_file_2, images, chunksize=1)
        return pd.DataFrame.from_records(results, columns=NAMES)  # type: ignore


def preprocess_file(class_: str, file: Path):
    image = cv2.imread(str(file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = subtract_background(image)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    folder = DATASET_PROCESSED_DIR / class_
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / file.name), result)


class SubtractBackground:
    def __init__(self) -> None:
        pass

    def __call__(self, image: Image) -> Image:
        return fromarray(subtract_background(np.array(image)))
