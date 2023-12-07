# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .userspace import UserspaceRTP

class SequenceList:
    """This class hold sequences as an ordered list and produces the metadata for each 
    sequence item."""
    
    def __init__(self, connections):
        """Return a new SequenceList object.""" 
        self.connections = connections
        self._seqitems = self.to_seqitems()

    def __iter__(self):
        for item in self._seqitems:
            yield item

    def to_seqitems(self):
        return [self._connection_to_seqitem(c) for c in self.connections]
     
    def _connection_to_seqitem(self, c):

        if isinstance(c.srckernel, UserspaceRTP):
            return {'name' : c.name,
                    'seqtype' : 'rtp',
                    'value' : c.srcport.value,
                    'srckernelname' : c.srcport.parent.name,             
                    'srcportname' : c.srcport.name, 
                    'snkkernelname' : c.sinkport.parent.name,   
                    'snkportname' : c.sinkport.name, 
                    }

        return {'name' : c.name,
            'seqtype' : 'buffer', 
            'offset' : c.srcport.offset,
            'nbytes' : c.srcport.nbytes,
            'srckernelname' : c.srcport.parent.name,             
            'srcportname' : c.srcport.name, 
            'srcdtype' : c.srcport.pdtype,
            'srcslices' : c.srcport.slices,
            'srcoffset' : c.srcport.offset,
            'srcnbytes' : c.srcport.nbytes,
            'srcshape' : c.srcport.shape,
            'snkkernelname' : c.sinkport.parent.name,   
            'snkportname' : c.sinkport.name, 
            'snkdtype' : c.sinkport.pdtype,
            'snkslices' : c.sinkport.slices,
            'snkoffset' : c.sinkport.offset,
            'snknbytes' : c.sinkport.nbytes,
            'snkshape' : c.sinkport.shape,            
        }
    
