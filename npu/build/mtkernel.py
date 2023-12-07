# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import numpy as np
from .tracekernels import kerneltracer
from .buffers import MTBuffer, Buffer
from .port import BufferPort
from .kernelmeta import KernelMeta


class MTKernel(KernelMeta):
    """This class is a superclass for all MemoryTile Kernels that run against Memory Tile buffers.
       These kernels can take an optional bufref to reuse an existing MT Buffer or by default
       create a new MT buffer when called.   

    Attributes
    ----------
    mtbuf : BufferPort
        The Memory Tile Buffer that is used within a MTKernel. 
    """
    def __init__(self, fxname):
        """Return a new MTKernel object.""" 
        self.mtbuf = None
        super().__init__(fxname, fxname, fxname, 'MT')

    def _validate_behavioral_call(self, bufref):
        if bufref is not None:        
            if not isinstance(bufref, np.ndarray):
                raise TypeError(f'MTKernel: mtbuf must be a ndarray for behavioral calls')

    def _validate_tracing_call(self, bufref):

        if bufref is not None:
            if not isinstance(bufref, BufferPort):
                raise TypeError(f'MTKernel: mtbuf must be a bufferport')
            if not isinstance(bufref.parent, MTBuffer):
                raise TypeError(f'MTKernel: Parent of mtbuf must be a MTBuffer')

        # if both self.mtbuf and mtbuf are set, but different - raise exception.
        if self.mtbuf is not None and bufref is not None and self.mtbuf is not bufref.parent:
            raise ValueError(f'Not possible to change the mtbuf variable in the middle of tracing...')


class MTPassThrough():
    """This class is used to wrap a MTPassThroughCall object using __new__.   

    Attributes
    ----------
    inputbuffer : BufferPort
        The buffer to be written into an MT Buffer.
    buferf: BufferPort
        Optional existing MT Buffer that is used for storing the incoming write.     
    """
    def __new__(self, inputbuffer=None) -> None:

        if inputbuffer is not None:
            return MTPassThroughCall()(inputbuffer)
        else:
            return MTPassThroughCall()


class MTPassThroughCall(MTKernel):
    """This class is used to execute the MT PassThrough kernel."""  

    def __init__(self):
        super().__init__('mtpassthrough')

    @kerneltracer
    def __call__(self, inputbuffer, behavioral_n_tracing=True):
        """ MT PassThrough is called.  Behaviorally, the input buffer is returned.  For 
            tracing, an MT Buffer is returned to track data movement"""
        if behavioral_n_tracing is True:
            return inputbuffer

        # tracing
        if self.mtbuf is None:
            self.mtbuf = MTBuffer(Buffer.to_ndarray(inputbuffer), mtmode='passthrough')
            self.mtbuf.createports(1,1)
        else:
            # self.mtbuf = bufref.parent
            self.mtbuf.array[:] = Buffer.to_ndarray(inputbuffer)    

        return self.mtbuf.outputbufferports[0]


class MTSplit():
    """This class is used to wrap a MTSplitCall object using __new__.   

    Attributes
    ----------
    inputbuffer : BufferPort
        The buffer to be written into an MT Buffer.
    numsplits : int
        The number of buffer splits requested to be returned.
    buferf: BufferPort
        Optional existing MT Buffer that is used for storing the incoming write.     
    """
    def __new__(self, *args) -> None:

        if len(args) == 1 and type(args[0]) is int:
            return MTSplitCall(args[0])
        elif len(args) == 2 and type(args[0]) in [BufferPort, np.ndarray] and type(args[1] is int):
            return MTSplitCall(args[1])(args[0])
        else:
            raise TypeError(f'MTSplit must be called with either BufferPort|ndarray, Int or Int as arguments')
            


class MTSplitCall(MTKernel):
    """This class is used to execute the MT Split kernel."""  
    def __init__(self, numsplits):
        self.numsplits = numsplits
        super().__init__('mtsplit')

    @kerneltracer
    def __call__(self, inputbuffer, behavioral_n_tracing=True):
        """ MT Split is called.  Behaviorally, the input buffer is split into numsplit arrays and 
        those arrays are returned.  For tracing, the correct number of output BufferPorts are created
        and are used in tracing of the application."""
        
        if behavioral_n_tracing is True:
            return np.split(inputbuffer, self.numsplits)

        # tracing 
        output_slices = _split_to_slices(inputbuffer.array, self.numsplits)

        if self.mtbuf is None:
            self.mtbuf = MTBuffer(Buffer.to_ndarray(inputbuffer), mtmode='split')

            portslices = [None] + [[os] for os in output_slices]
            self.mtbuf.createports(1,self.numsplits, portslices)
        else:
            self.mtbuf.array[:] = Buffer.to_ndarray(inputbuffer)

        return self.mtbuf.outputbufferports 


class MTConcat():
    """This class is used to wrap a MTConcatCall object using __new__.   

    Attributes
    ----------
    inputbuffers : List
        The list of bufferPorts to be concatenated.
    buferf: BufferPort
        Optional existing MT Buffer that is used for storing the incoming BufferPorts.     
    """
    def __new__(self, inputbuffers=None) -> None:

        if inputbuffers is not None:
            return MTConcatCall()(inputbuffers)
        else:
            return MTConcatCall()


class MTConcatCall(MTKernel):
    """This class is used to execute the MT Concat kernel."""  
    def __init__(self) -> None:
        super().__init__('mtconcat')

    @kerneltracer
    def __call__(self, inputbuffers, behavioral_n_tracing=True):
        """ MT Concat is called.  Behaviorally, the input buffers are concatenated into a single array and 
        that final array is returned.  For tracing, the correct number of input BufferPorts are created
        and are used in tracing of the application."""
        ibs = [Buffer.to_ndarray(ib) for ib in inputbuffers]
        ndconcat = np.concatenate(ibs)

        if behavioral_n_tracing is True:
            return ndconcat

        # tracing
        if self.mtbuf is None:
            portslices = [ [s] for s in _concat_to_slices(ibs)] + [None]
            self.mtbuf = MTBuffer(ndconcat, mtmode='concat')
            self.mtbuf.createports(len(inputbuffers),1, portslices)
        else:
            self.mtbuf.array[:] = ndconcat

        return self.mtbuf.outputbufferports[0]


def _split_to_slices(a, numsplits):
    """Return the Slice objects for use in split operations."""
    if not isinstance(numsplits, int):
        raise TypeError(f'can only split an array using an integer split of axis=0')
    
    nsections, remaining = divmod(a.shape[0], numsplits)        
    if remaining:
        raise ValueError('can only split an array evenly')    
                        
    divpoints=[nsections*i for i in range(numsplits)] + [len(a)]
    return _divpoints_to_slices(divpoints)     


def _concat_to_slices(inputbuffers):
    """Return the Slice objects for use in concatenation operations."""
    input_lengths = np.array([0] + [len(ib) for ib in inputbuffers])
    divpoints = np.cumsum(input_lengths)    
    return _divpoints_to_slices(divpoints)


def _divpoints_to_slices(divpoints):
    slices = list()
    for i in range(len(divpoints)-1):
        if divpoints[i+1] - divpoints[i] == 1:
            slices.append(divpoints[i])
        else:
            slices.append(slice(divpoints[i],divpoints[i+1]))

    return slices    
