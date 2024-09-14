# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import platform
from .xbutil import XBUtil


def nputop():
    """ Uses XBUtil to display all currently running applications in an ipywidgets
    form suitable for JupyterLab """
    if platform.system() == "Windows":
        XBUtil().apps()
    else:
        print("nputop is not currently supported in linux due to changes in the xbutil api")
