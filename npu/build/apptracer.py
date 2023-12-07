# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np

from .connections import Connection
from .tracelogger import TraceLogger
from .buffers import MTBuffer, ITBuffer
from .port import BufferPort
from .kernel import Kernel
from .mtkernel import MTKernel
from .itkernel import ITKernel, ITWrite

class AppTracer():
    """This class manages the tracing of an AppBuilder's callgraph method to determine:
    1. unique kernels used in the callgraph
    2. unique connections between IT, MT, and CT kernels
    3. ordered sequence of data movements between kernels using connections

    Attributes
    ----------
    application : AppBuilder
        The AppBuilder instance passed in with a valid callgraph for tracing

    """
    def __init__(self, application):
        """Return a new AppTracer object."""
        self.application = application       

    def to_trace(self, *args):
        """Using the TraceLogger class, the application's callgraph is traced 
           and returns discovered kernels and connections.
        """
        TraceLogger.clear_trace_objects()
        TraceLogger.is_traceenabled = True

        args = [ITBuffer.ndarray_to_itb_port(a, setitem_itkernel=ITWrite) for a in args]
        trace_retval = self.application.callgraph(*args)

        tkernels, tconnections = self._postprocess_trace(trace_retval)   
        TraceLogger.is_traceenabled = False          

        return tkernels, tconnections

    def _postprocess_trace(self, trace_retval):
        """ Traced input buffers, output buffers and kernel calls are processed 
            to discover buffer ports, RTP values and ComputeTile kernel arguments.
        """
        trace_kernels = list()
        trace_connections = list()
        for kernel, args, _, result in TraceLogger.trace_kernelcalls:

            kernel = self.postprocess_kernel(kernel, result)
            buffer_args, rtp_args, kernels_in_args = self.postprocess_args(kernel, args)

            if len(buffer_args) != len(kernel.inputbufferports):
                raise ValueError(f'Kernel {kernel.name} is called with {len(buffer_args)} args, expecting {len(kernel.inputbufferports)} args ')

            trace_kernels.extend([kernel] + kernels_in_args)            
            trace_connections.extend([Connection(c) for c in zip(buffer_args, kernel.inputbufferports)])
            trace_connections.extend([Connection(c) for c in zip(rtp_args, kernel.rtpports)]) 


        # append kernels and connections based on return type of the callgraph()
        if trace_retval:
            if not isinstance(trace_retval, tuple):
                trace_retval = (trace_retval,)

            for t in trace_retval:
                if isinstance(t.parent, ITBuffer):
                    continue

                ub = ITBuffer(t.array)
                trace_kernels.append(ub)
                trace_connections.append(Connection((t,ub)))


        return trace_kernels, trace_connections

    def postprocess_kernel(self, k, res):
        """ Kernel type specific postprocessing to get the right object for tracing."""
        if isinstance(k, Kernel):
            return k
        
        if isinstance(k, MTKernel):
            return k.mtbuf
        
        if isinstance(k, ITKernel):
            return k.itbuf        
        
        raise TypeError(f'Unable to trace Kernel of type {type(k)}')

    def postprocess_args(self, kernel, args):
        """ Postprocess args to get buffer ports, RTP values and any additional kernels that
            are passed into functions directly.
        """

        # args as list need broken out for analysis
        args = self._flatten_list(args)

        # MTKernels and ITKernels pass in optional bufref port that shouldn't be traced
        args = self._untrace_bufref_ports(kernel, args)

        # ndarrays converted to an ITBuffer output port for tracing
        args = self._convert_ndarrays(args)

        buffer_args = [a for a in args if isinstance(a, BufferPort)]
        kernels_in_args = [a.parent for a in buffer_args]
        rtp_args = [a for a in args if isinstance(a, int)]
        return buffer_args, rtp_args, kernels_in_args
    
    def _untrace_bufref_ports(self, kernel, args):
        """IT and MTKernels with bufref optional argument require removal of that bufref 
           since the bufref is not an input or output argument of the kernel.
        """
        if type(kernel) not in [MTBuffer, ITBuffer]:
            return args
        
        buffer_args = [a for a in args if isinstance(a, BufferPort)]
        if len(buffer_args) == len(kernel.inputbufferports):
            return args
        
        if len(buffer_args) == len(kernel.inputbufferports) + 1:
            return args[:-1]        

        raise ValueError(f'Unable to trace Kernel of type {type(kernel)}')        

    def _flatten_list(self, args):
        newargs = list()
        for a in args:
            _ = newargs.extend(a) if isinstance(a, list) else newargs.append(a)
        return newargs

    def _convert_ndarrays(self, args):
        for a in args:
            if isinstance(a, np.ndarray):
                raise ValueError(f'Should not be tracing ndarrays - they should be converted at callgraph arg parse') 

        return [ITBuffer.ndarray_to_itb_port(a) for a in args]
