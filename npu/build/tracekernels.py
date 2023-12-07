# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import functools
import inspect
from .tracelogger import TraceLogger
from copy import deepcopy

def kerneltracer(func):
    """Wrapper around traced functions to enable logging of calls in the global TraceLogger."""
    @functools.wraps(func)
    def graphtrace(*args, **kwargs):

        if isfunctraceable(func):
            kwargs['behavioral_n_tracing'] = not TraceLogger.is_traceenabled

        result = func(*args, **kwargs)

        # Update Traced Objects (funcname, func_args, func_kwargs)
        newkernelcall = (deepcopy(args[0]), args[1:], kwargs, result)

        TraceLogger.update_trace_objects(newkernelcall)
    
        return result

    return graphtrace



def isfunctraceable(func):
    sig = inspect.signature(func)
    return any([p.name == 'behavioral_n_tracing' for p in sig.parameters.values()])
