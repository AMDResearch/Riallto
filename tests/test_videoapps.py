# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from npu.utils import videoapps
from npu.lib import (
    ColorDetectVideoProcessing, ColorThresholdVideoProcessing,
    DenoiseDPVideoProcessing, DenoiseTPVideoProcessing,
    EdgeDetectVideoProcessing, ScaledColorThresholdVideoProcessing
)


files = ['../notebooks/images/jpg/toucan.jpg',
         '../notebooks/images/png/ryzen-ai-sdk.png',
         '../notebooks/images/gif/ping_pong_buffer.gif']

testcases = [(vapp, file) for file in files for vapp in videoapps()]


@pytest.mark.parametrize('testcase', testcases)
def test_videoapp_use_jpg(testcase):
    app, filename = testcase
    appobj = eval(app)(filename)
    assert appobj
    del appobj
