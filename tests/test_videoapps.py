# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import os
from npu.lib import (
    ColorDetectVideoProcessing, ColorThresholdVideoProcessing,
    EdgeDetectVideoProcessing, ScaledColorThresholdVideoProcessing,
    DenoiseDPVideoProcessing, DenoiseTPVideoProcessing
)

apps = ['ColorDetectVideoProcessing', 'ColorThresholdVideoProcessing',
        'EdgeDetectVideoProcessing', 'ScaledColorThresholdVideoProcessing',
        'DenoiseDPVideoProcessing', 'DenoiseTPVideoProcessing']

files = ['../notebooks/images/jpg/toucan.jpg',
         '../notebooks/images/png/ryzen-ai-sdk.png',
         '../notebooks/images/gif/ping_pong_buffer.gif']

testcases = [f'{vapp}; {file}' for file in files for vapp in apps]


@pytest.mark.parametrize('testcase', testcases)
def test_videoapp_use_jpg(testcase):
    app, filename = testcase.split(';')
    filename = os.path.dirname(os.path.abspath(__file__)) +'/'+ filename
    appobj = eval(app)(filename)
    assert appobj
    del appobj
