# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from .utils import is_win

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
BUILD_TEMPLATE_PATH = os.path.join(MODULE_PATH,"build_template")

def wslpath(winpath:str)->str:
    """ From the windows path create the equivalent WSL path """
    if is_win():
        drive = winpath[0].lower()
        tpath = winpath[3:].replace("\\", "/")
        return f"/mnt/{drive}/{tpath}"
    else:
        return winpath

