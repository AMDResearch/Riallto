# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import cv2
from pathlib import Path
import numpy as np
from .kernelgenerator import KernelObjCall


class Gray2Rgba():
    """Vectorized implementation of the `cv2.cvtColor` RGBA2Gray conversion.

    This kernel is used to convert a single channel grayscale image input
    into a 4 channel RGBA image.

    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "gray2rgba.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        rgba = cv2.cvtColor(self.in_buffer.array, cv2.COLOR_GRAY2RGBA)
        self.out_buffer.array = np.squeeze(rgba)


class Gray2RgbaScalar():
    """Scalar implementation of the `cv2.cvtColor` RGBA2Gray conversion."""

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "gray2rgba_scalar.cpp")
        return KernelObjCall(cpp, Gray2Rgba.behavioralfx, *args)
