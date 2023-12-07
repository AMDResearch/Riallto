# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Applications
------------

Prebuilt applications can be run as-is with no additional build
steps required. The video applications include a variety of computer
vision pipelines like color threshold or edge detect.

You can list the available videoapps with

    from npu.utils import videoapps
    videoapps()

Example running edge detect

    from npu.lib import EdgeDetectVideoProcessing

    app = EdgeDetectVideoProcessing()
    app.start()


"""

from .videoapps import pxtype
from .videoapps import VideoApplication
from .videoapps import ColorThresholdVideoProcessing
from .videoapps import ScaledColorThresholdVideoProcessing
from .videoapps import EdgeDetectVideoProcessing
from .videoapps import ColorDetectVideoProcessing
from .videoapps import DenoiseTPVideoProcessing
from .videoapps import DenoiseDPVideoProcessing
