# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from .kernelgenerator import KernelObjCall


class ThresholdRgba():
    """Vectorized implementation of the `cv2.threshold` function.

    This vectorized implementation operates on RGBA image inputs.
    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    """
    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "threshold_rgba.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = self.in_buffer.array

        # TODO verify and map different threshold types into the cv2 threshold type enum
        # outImageReference = cv2.Mat(self.in1.array.shape(0), self.in1.array.shape(1), cv2.CV_8UC1)
        # self.out1.array = cv2.threshold(self.in1.array, outImageReference, 100,255,cv2.THRESH_BINARY)


class ThresholdGrayscale():
    """Vectorized implementation of the `cv2.threshold` function.

    This vectorized implementation operates on RGBA image inputs.
    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    """
    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "threshold_grayscale.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = self.in_buffer.array

        # TODO verify and map different threshold types into the cv2 threshold type enum
        # outImageReference = cv2.Mat(self.in1.array.shape(0), self.in1.array.shape(1), cv2.CV_8UC1)
        # self.out1.array = cv2.threshold(self.in1.array, outImageReference, 100,255,cv2.THRESH_BINARY)


class RgbaRtpThres():
    """Vectorized implementation of an RGBA threshold.

     For each RGBA 4 byte pair it applies a threshold to the R/G/B channels
     while leaving the alpha channel unchanged. This kernel accepts one
     buffer input for the input tile size, a RTP nbytes  for the number of
     bytes that the kernel is expecting to process, r_thresh the threshold
     value for the red channel, g_thres the threshold value for the green
     channel, and b_thresh the threshold value for the blue channel.
     The vectorization factor for the kernel is 64 elements.
     """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "rgba_rtp_thresh.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = self.in_buffer.array
