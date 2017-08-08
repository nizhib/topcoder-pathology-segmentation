import math
import random

import cv2
from PIL import Image

from abstract import DualTransformer


class DualPad(DualTransformer):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def transform(self, img, how):
        p = self.padding
        return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT_101)


def rotate_padded(image, angle, padding):
    orig_shape = image.shape

    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError('Too lazy to work with non-even sizes')

    max_shape = max(image.shape[0], image.shape[1])
    px = int(math.ceil(0.25 * (2 * max_shape - image.shape[1] + 2 * padding)))
    py = int(math.ceil(0.25 * (2 * max_shape - image.shape[0] + 2 * padding)))
    image = cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_REFLECT_101)
    cols, rows = image.shape[:2]

    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (rows, cols), cv2.INTER_CUBIC)

    dy = (rotated.shape[0] - orig_shape[0]) // 2 - padding
    dx = (rotated.shape[1] - orig_shape[1]) // 2 - padding

    result = rotated[dy:-dy, dx:-dx].copy()

    return result


def rotate_crop_padded(image, sx, sy, angle, size):
    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError('Too lazy to work with non-even sizes')

    max_shape = max(image.shape[0], image.shape[1])
    px = py = int(max(0.0, math.ceil(3 * size - max_shape / 2)))
    image = cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_REFLECT_101)
    cols, rows = image.shape[:2]

    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (rows, cols), cv2.INTER_CUBIC)

    dy = (rotated.shape[0] - size) // 2
    dx = (rotated.shape[1] - size) // 2

    result = rotated[dy + sy:-dy + sy, dx + sx:-dx + sx].copy()

    return result


class DualRotatePadded(DualTransformer):
    def __init__(self, max_angle, padding=6):
        super().__init__()
        self.max_angle = max_angle
        self.padding = padding
        self.angle = 0.0

    def transform(self, img, how):
        if how == 'image':
            self.angle = random.uniform(-self.max_angle, self.max_angle)
        return rotate_padded(img, self.angle, self.padding)


class DualRotateCropPadded(DualTransformer):
    def __init__(self, max_angle, max_shift, size=512):
        super().__init__()
        self.max_angle = max_angle
        self.max_shift = max_shift
        self.size = size
        self.angle = 0.0
        self.sx = 0.0
        self.sy = 0.0

    def transform(self, img, how):
        if how == 'image':
            self.angle = random.uniform(-self.max_angle, self.max_angle)
            self.sx = int(random.uniform(-self.max_shift, self.max_shift))
            self.sy = int(random.uniform(-self.max_shift, self.max_shift))
        return rotate_crop_padded(img, self.sx, self.sy, self.angle, self.size)


class DualToPIL(DualTransformer):
    def transform(self, img, how):
        if how == 'image':
            return Image.fromarray(img[:, :, ::-1])
        else:
            return Image.fromarray(img)
