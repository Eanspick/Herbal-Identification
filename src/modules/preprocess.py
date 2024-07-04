import cv2
import numpy as np
from cv2.typing import MatLike
from PIL.Image import Image, fromarray

KERNEL = np.ones((50, 50), np.uint8)


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


def subtract_background(image: MatLike):
    height, width, _ = image.shape
    image = cv2.resize(image, (2000, 2000))
    black_img = np.empty((2000, 2000, 3), dtype=np.uint8)

    contour = find_contour(image)
    mask = cv2.drawContours(black_img, [contour], 0, (255, 255, 255), -1)
    masked_img = cv2.bitwise_and(image, mask)
    final_img = np.where(masked_img.any(-1, keepdims=True), masked_img, 255)
    return cv2.resize(final_img, (height, width))


class SubtractBackground:
    def __init__(self) -> None:
        pass

    def __call__(self, image: Image) -> Image:
        return fromarray(subtract_background(np.array(image)))
