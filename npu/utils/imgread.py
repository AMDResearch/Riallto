# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import cv2
import numpy as np

_resolution = (720, 1280)

class OpenCVImageReader:
    """Read an image file using OpenCV and return it as a NumPy ndarray

    This reader is constrained to images with resolution 720p (720, 1280).
    You can read images with different resolutions by setting:
    `any_resolution=True`
    """
    def __init__(self, filename: str, grayscale: bool = False,
                 any_resolution: bool = False) -> None:
        """
        Parameters
        ----------
        filename: image file path
        grayscale: return the image using grayscale color space
        any_resolution: read images of any resolution
        """
        img = cv2.imread(filename)
        if not any_resolution and (img.shape[0], img.shape[1]) != _resolution:
            raise ValueError(f"Image shape (({img.shape[0]},{img.shape[1]})) "
                             "is not compatible. Supported resolution: "
                             f"{_resolution}. You can read an image of "
                             "any resolution by setting "
                             "`any_resolution=True`")
        if grayscale:
            typecolor = cv2.COLOR_BGR2GRAY
        else:
            typecolor = cv2.COLOR_BGR2RGBA

        self._img = cv2.cvtColor(img, typecolor)

    @property
    def img(self) -> np.ndarray:
        """Returns image in ndarray format"""
        return self._img
