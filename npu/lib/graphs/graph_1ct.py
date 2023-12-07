# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from typing import Optional
from pathlib import Path
import inspect

from npu.build.appbuilder import AppBuilder
from npu.build.mtkernel import MTPassThrough
from npu.build.itkernel import ITWrite
from npu.build.kernel import Kernel
from npu.build.port import BufferPort, RTPPort

def _graph_1ct_720_row_default_behavioural(invobj):
    """ Default behavioral model for single kernel in RGB720pBuilder AppBuilder class"""
    invobj.out_buffer.array = invobj.in_buffer.array

class RGB720pBuilder(AppBuilder):
    """ An Application builder that assigns a single kernel to a CT with the following:
    * Restriction on signature that it has only 1 input buffer called in_buffer.
    * Restriction on signature that it has only 1 output buffer called out_buffer.
    * Kernel can have as many RTP as it likes. 
    * Restriction that the shape for in_buffer and out_buffer are (720,1280,4) with dtype np.uint8
    
    Attributes
    ----------
    _kernel : npu.build.Kernel
        The kernel to be used with this graph
    _kports : Set[str]
        The expected set of buffers with this kernel
    _ktype : str
        The expected c-type for the buffer ports
    _kport_shape : tuple[int,int,int]
        The expected shape for the buffers
    _rtps : list[int]
        The initial values for all RTPs in the input kernel
    img_in : np.array
        A numpy array used for application tracing with the default behavioral model for the kernel.
    img_out : np.array
        A numpy array used for application tracing with the default behavioral model for the kernel.
    """ 

    def __init__(self, kernel:Kernel)->None:
        """ A Write-your-own kernel application
        Builds a graph for a restricted kernel types.
        """
        # Getting the name of the variable
        frame = inspect.currentframe().f_back
        varnames = frame.f_locals.keys()

        self._kernel = kernel 

        self._kports = {"in_buffer", "out_buffer"}
        self._ktype = "uint8_t *"
        self._kport_shape = (720, 1280, 4) 
        
        self._kernel.behavioralfx = _graph_1ct_720_row_default_behavioural
        self._kernel_crude_signature_check(self._kernel)
        self._kernel.tloc = (0,2)
        super().__init__(name=kernel.shortname)
        self._rtps = [128] * len(kernel.rtpports) 

        img_in = np.zeros(shape=(720,1280,4), dtype=np.uint8)
        img_out = np.zeros(shape=(720,1280,4), dtype=np.uint8)
        _ = self.to_json(img_in, img_out)

    def _kernel_crude_signature_check(self, k:Kernel)->None:
        """ ensure that the type signature of the kernel is
        valid for this application.
        """
        seen = set() 
        rtps = set()
        for p in k.ports:
            if isinstance(p, BufferPort):
                seen.add(p.name)
                if not p.name in self._kports:
                    raise RuntimeError(f"Kernel function args are restricted to only allow the following bufferports: {self._kports} port {p.name} is not permitted")

                if p.pdtype != self._ktype:
                    raise RuntimeError(f"Kernel port with type {self._ktype} is only allowed with this type of graph, port {p.pdtype} is forbidden.")

            if isinstance(p, RTPPort):
                rtps.add(p.name)

        if seen != self._kports:
            raise RuntimeError(f"Kernel with buffer ports {seen} but expecting {self._kports}")

        if "nbytes" not in rtps:
            raise RuntimeError(f"Expecting RTP port nbytes in list of rtps to specify amount of data kernel needs to process, only found {rtps=}")
        
    def _check_inout_shape(self, x_in:np.ndarray, x_out:np.ndarray)->None:
        """ Raises an exception if the buffer shapes are not what are expected"""
        if x_in.shape != self._kport_shape:
            raise RuntimeError(f"input port has shape {x_in.shape=} expecting shape {self._kport_shape}")
        if x_out.shape != self._kport_shape:
            raise RuntimeError(f"output port has shape {x_out.shape=} expecting shape {self._kport_shape}")

    def callgraph(self, x_in, x_out):
        """ Callgraph that checks that builds a single kernel callgraph with the IT with restrictions. """
        self._check_inout_shape(x_in, x_out)
        self._rtps[0] = 1280*4 
        for i in range(720):
            y = self._kernel(x_in[i], *self._rtps)
            _ = ITWrite(y, bufref=x_out[i])
        return


        
