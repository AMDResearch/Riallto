# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import inspect
import npu.lib.applications


def _is_subclass(member):
    return inspect.isclass(member) and \
        issubclass(member, npu.lib.applications.VideoApplication)


def videoapps():
    """Returns a list of pre-made video processing applications"""
    return [x[0] for x in
            inspect.getmembers(npu.lib.applications,
                               predicate=_is_subclass)]
