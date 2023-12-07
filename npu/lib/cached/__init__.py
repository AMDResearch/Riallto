# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Cached
======

In order to avoid recompiling identical kernels a caching
mechanism is implemented and all compiled kernels are stored
in the `npu.lib.cached` directory.

When compiling a previously generated kernel .so object, an
md5 checksum is performed on the source code and if it is
identical the cached kernel will be copied instead of being
recompiled.
"""
