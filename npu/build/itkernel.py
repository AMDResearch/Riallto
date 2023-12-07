# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


from .tracekernels import kerneltracer
from .buffers import ITBuffer, Buffer
from .kernelmeta import KernelMeta
from .port import BufferPort

class ITKernel(KernelMeta):
    """This class is a superclass for all Kernels that run against Interface Tile buffers.
       These kernels can take an optional bufref to reuse an existing IT Buffer or by default
       create a new IT buffer when called.   

    Attributes
    ----------
    itbuf : BufferPort
        The Interface Tile Buffer that is used within a ITKernel. 
    """

    def __init__(self, fxname):
        """Return a new ITKernel object.""" 
        self.itbuf = None
        super().__init__(fxname, fxname, fxname, 'IT')

    def _validate_trace_call(self, bufref):
        """ verify the bufref has the right type and is not replaced erroneously during an application trace"""
        if bufref is None:
            return

        if not isinstance(bufref, BufferPort):
            raise TypeError(f'bufref of type {type(bufref)} is not a BufferPort')

        if not isinstance(bufref.parent, ITBuffer):
            raise TypeError(f'Parent of bufref of type {type(bufref.parent)} is not an ITBuffer')

        # if both self.itbuf and itbuf are set, but different - raise exception.
        if self.itbuf is not None and bufref is not None and self.itbuf.name != bufref.parent.name:
            raise ValueError(f'Not possible to change the itbuf variable in the middle of tracing...')

class ITWrite():
    """This class is used to wrap a ITWriteCall object using __new__.   

    Attributes
    ----------
    inputbuffer : BufferPort
        The buffer to be written into an IT Buffer.
    buferf: BufferPort
        Optional existing IT Buffer that is used for storing the incoming write.     
    """
    def __new__(self, inputbuffer=None, bufref=None) -> None:
        if inputbuffer is not None:
            return ITWriteCall()(inputbuffer, bufref)
        else:
            return ITWriteCall()


class ITWriteCall(ITKernel):
    """This class is used to execute the IT Write kernel."""   
    
    def __init__(self):
        super().__init__('itwrite')

    def _validate_itwrite_trace_call(self, inputbuffer):
        if isinstance(inputbuffer.parent, ITBuffer):
            raise TypeError(f'ITWrite can only be written by CT or MT bufferports.  inputbuffer.parent is of type: {type(inputbuffer.parent)}')
        
    @kerneltracer
    def __call__(self, inputbuffer, bufref=None, behavioral_n_tracing=True):
        """ IT Write is called.  Behaviorally, the input buffer is returned.  For 
            tracing, an IT Buffer is returned to track data movement"""
        if behavioral_n_tracing is True:
            if bufref is not None:
                bufref[:] = inputbuffer
               
            return inputbuffer
        self._validate_itwrite_trace_call(inputbuffer)
        self._validate_trace_call(bufref)

        # tracing
        if bufref is None and self.itbuf is None:
            self.itbuf = ITBuffer(Buffer.to_ndarray(inputbuffer))
        else:
            # construct itb from incoming itb.bufferport (bufref)
            self.itbuf = bufref.parent
            self.itbuf.inputbufferports[0].slices = bufref.slices
            bufref.array[:] = Buffer.to_ndarray(inputbuffer).reshape(bufref.array.shape)  
        
        return self.itbuf.outputbufferports[0]


class ITRead():
    """This class is used to implement an IT Read Call.   

    Attributes
    ----------
    inputbuffer : BufferPort
        The buffer that is to be read.   
    """
    def __new__(self, inputbuffer=None) -> None:
        if inputbuffer is not None:
            return ITReadCall()(inputbuffer)
        else:
            return ITReadCall()


class ITReadCall(ITKernel):
    """This class is used to execute the IT Read kernel."""   
    
    def __init__(self):
        super().__init__('itread')

    def _validate_itread_trace_call(self, inputbuffer):
        if not isinstance(inputbuffer.parent, ITBuffer):
            raise TypeError(f'ITRead can only be written by IT bufferports.  inputbuffer.parent is of type: {type(inputbuffer.parent)}')
        
    def __call__(self, inputbuffer, behavioral_n_tracing=True):
        """ IT Read is called.  Behaviorally, the input buffer is returned.  For 
            tracing, the inputbuffer is also returned.  Note that ITReads are not traced
            using the @kerneltracer decorator - ITRead usgae is mostly for code readability."""
        if behavioral_n_tracing is True:            
            return inputbuffer
        
        self._validate_itread_trace_call(inputbuffer)
        return inputbuffer.outputbufferports[0]
