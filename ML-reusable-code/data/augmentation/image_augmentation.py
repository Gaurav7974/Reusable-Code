import cv2
import numpy as np
from typing import Tuple


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by angle degrees.
    """
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def flip(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    """
    Flip image horizontally or vertically.
    """
    return cv2.flip(image, 1 if horizontal else 0)


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image.
    """
    return cv2.resize(image, size)
