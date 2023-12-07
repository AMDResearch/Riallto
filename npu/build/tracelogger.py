# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .kernelmeta import KernelMeta

class TraceLogger():
    """Global class used for tracing an AppBuilder callgraph."""
    trace_kernelcalls = list()
    is_traceenabled = False

    @classmethod
    def update_trace_objects(cls, newkernelcall):
        cls.trace_kernelcalls.append(newkernelcall)     

    @classmethod
    def clear_trace_objects(cls):
        cls.trace_kernelcalls = list()
        KernelMeta.reset_unique_names()

    def __init__(self):
        pass