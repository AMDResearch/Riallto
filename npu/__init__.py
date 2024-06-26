# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
NPU
=====

Documentation is available in the docstrings and online at https://riallto.ai.

Provides
  1. An easy-to-use runtime library to run custom applications on the NPU.
  2. APIs for building custom applications.
  3. Useful utilities to test and introspect your designs.

Available subpackages
---------------------
build
    Tools for building applications.
lib
    Libraries of prebuilt kernels and graphs.
runtime
    Runtime libraries based on XRT.
utils
    Utilities for building, testing and visualization.

"""

from .utils.test_device import get_driver_version, version_to_tuple
import platform

__supported_driver__ = "10.1109.8.100"

if platform.system() == 'Windows':
    __installed_driver__ = get_driver_version()
    
    if version_to_tuple(__installed_driver__) < version_to_tuple(__supported_driver__):
        raise ValueError(f"""Detected driver: {__installed_driver__}, supported driver version is >={__supported_driver__},
                  go to https://riallto.ai/prerequisites-driver.html for driver setup instructions.""")

from .repr_dict import ReprDict
from .magic import kernel_magic
from .utils.nputop import nputop
