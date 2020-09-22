import os
from pathlib import Path
import numpy as np

from image_sample import ImageSample


class SimpleDataset:
    def __init__(self, dataset_dir_path: str):
        self._dataset_dir_path = dataset_dir_path

    def __call__(self):
        images_path = os.path.join(self._dataset_dir_path, "images")
        masks_path = os.path.join(self._dataset_dir_path, "masks")

        images = [Path(os.path.join(images_path, x)) for x in os.listdir(images_path)]
        masks = [Path(os.path.join(masks_path, x)) for x in os.listdir(masks_path)]

        for image_path, mask_path in zip(images, masks):
            image = ImageSample(image_path).get_image_bytes()
            image = image.astype(np.float64) / 255
            image = np.expand_dims(image, axis=2)

            mask = ImageSample(mask_path).get_image_bytes()
            mask = mask.astype(np.float64) / 255
            mask = np.expand_dims(mask, axis=2)

            masks = np.concatenate((1 - mask, mask), axis=2)

            yield image, masks