# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from .test_applications import manage_testing
from .test_applications import AppITTilingMTTilingRgbaRtpThres


def test_linkop(manage_testing):
    """ Build test that uses the memtile to test MLIR linkop generation. """
    img_w = 1280
    img_h = 720

    imgbuffer_in = np.zeros(shape=(img_h,4,img_w),dtype=np.uint8)
    imgbuffer_out = np.zeros(shape=(img_h,4,img_w),dtype=np.uint8)

    r_thresh = [255, 0, 255, 101]
    g_thresh = [193, 255, 255, 0]
    b_thresh = [0, 0, 255, 255]

    tilesize = img_w

    trace_app = AppITTilingMTTilingRgbaRtpThres()
    trace_app.build(imgbuffer_in,imgbuffer_out,tilesize,r_thresh,g_thresh,b_thresh)
