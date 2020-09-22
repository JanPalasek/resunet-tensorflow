from pathlib import Path
import cv2
import numpy as np


class ImageSample:
    def __init__(self, image_path: Path):
        self._image_path = image_path
        self._bytes = None

    @property
    def image_path(self):
        return str(self._image_path)

    @property
    def image_stem(self):
        return str(self._image_path.stem)

    @property
    def image_name(self):
        return str(self._image_path.name)

    def get_image_bytes(self) -> np.ndarray:
        if self._bytes is None:
            self._bytes = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        return self._bytes