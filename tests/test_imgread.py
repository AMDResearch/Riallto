# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import pytest
import warnings
from npu.utils import OpenCVImageReader, image_plot

_ryzenai = 'notebooks/images/jpg/ryzenai_future_starts_now.jpg'
_toucan = 'notebooks/images/jpg/toucan.jpg'
_png = 'notebooks/images/png/AMD-Ryzen-AI-Main-1.png'


@pytest.mark.skipif(not os.path.exists(_ryzenai), reason='Image not found')
def test_rgba_image():
    """ Tests reading an RGBA image with the OpenCVImageReader class"""
    img = OpenCVImageReader(_ryzenai).img
    try:
        image_plot(img)
    except Exception as e:
        warnings.warn(f'Exception occurred: {e}')
    assert img.any()


@pytest.mark.skipif(not os.path.exists(_ryzenai), reason='Image not found')
def test_grascale_image():
    """ Tests reading an RGBA image with the OpenCVImageReader class"""
    img = OpenCVImageReader(_ryzenai, True).img
    try:
        image_plot(img)
    except Exception as e:
        warnings.warn(f'Exception occurred: {e}')
    assert img.any()


@pytest.mark.skipif(not os.path.exists(_toucan), reason='Image not found')
def test_any_resolution_jpg_image():
    """Tests reading an RGBA image using standard resolution of the OpenCVImageReader class"""
    img = OpenCVImageReader(_toucan, any_resolution=True).img
    try:
        image_plot(img)
    except Exception as e:
        warnings.warn(f'Exception occurred: {e}')
    assert img.any()


@pytest.mark.skipif(not os.path.exists(_png), reason='Image not found')
def test_any_resolution_png_image():
    """Tests reading an RGBA image using standard resolution of the OpenCVImageReader class"""
    img = OpenCVImageReader(_png, any_resolution=True).img
    try:
        image_plot(img)
    except Exception as e:
        warnings.warn(f'Exception occurred: {e}')
    assert img.any()


@pytest.mark.skipif(not os.path.exists(_toucan), reason='Image not found')
def test_unsopported_resolution():
    """Tests reading an RGBA image not using standard resolution of the OpenCVImageReader class"""

    with pytest.raises(ValueError) as excinfo:
        _ = OpenCVImageReader(_toucan)

    assert "is not compatible. Supported resolution: " in str(excinfo.value)
