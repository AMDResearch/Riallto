# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from .buffers import Buffer

class AppMetada:
    """This class contains the in-memory representation of the AppBuilder application and
    completes any unspecified application placements (E.g. ComputeTile kernel mapping).

    Attributes
    ----------
    appname : str
        The name of the application.
    kernels : dict
        Dictionary storing all unique compute tile kernels in this application.
    connections : dict
        Dictionary storing all unique connections between kernels in this application.
    sequence : list
        List containing the ordered data movements as traced in the AppBuilder's callgraph.

    """

    def __init__(self, appname, kernels, connections, sequence):
        """Return a new AppMetada object."""
        self.appname = appname
        self.kernels = kernels
        self.connections = connections
        self.sequence = sequence

        self._complete_unconstrained_spec()


    def _complete_unconstrained_spec(self):
        """ Verify the given constaints and complete unspecified constraints
            found in the application.
        """

        # Place unplaced CT kernels get assigned to a tile
        ct_tiles = [(0,ix) for ix in range(2,6)]
        ct_kernels = [k for k in self.kernels if k.ttype == 'CT']
        

        self._verify_constrained_cttiles(ct_tiles)
        self._place_unconstrainted_kernels(ct_tiles) 

        if None in [k.tloc for k in ct_kernels]:
            raise ValueError(f'Unable to place all kernels at CT tiles')

    def _verify_constrained_cttiles(self, ct_tiles):
        ct_kernels_tloc = [k for k in self.kernels if k.ttype == 'CT' and k.tloc is not None]
        for k in ct_kernels_tloc:
            if k.tloc not in ct_tiles:
                raise ValueError(f'Cannot place kernel {k.name} at {k.tloc} - {k.tloc} does not exist')

    def _place_unconstrainted_kernels(self, ct_tiles):
        """ ComputeTile kernels are mapped onto open tiles."""
        ct_kernels_no_tloc = [k for k in self.kernels if k.ttype == 'CT' and k.tloc is None]

        for k in ct_kernels_no_tloc:

            used_ct_tiles = [k.tloc for k in self.kernels if k.ttype == 'CT' and k.tloc is not None]
            free_ct_tiles = list(set(ct_tiles) - set(used_ct_tiles))

            if len(free_ct_tiles) == 0:
                raise ValueError(f'Unable to place kernel {k.name} - no more free CT tiles')
            
            k.tloc = free_ct_tiles[0]

    def to_json(self):
        _json = OrderedDict()
        _json['application'] = self.appname
        _json['kernels'] = OrderedDict()
        for k in self.kernels:
            _json['kernels'][k.name] = k.to_metadata() 

        _json['connections'] = OrderedDict()        
        for c in self.connections:            
            cdict = {'name'  : c.name,
                     'srckernel' : c.srckernel.name,
                     'srcport' : c.srcport.name,
                     'sinkkernel' : c.sinkkernel.name,
                     'sinkport' : c.sinkport.name,
                     'dtype' : c.dtype,
                     'ctype' : c.ctype}
            _json['connections'][c.name] = cdict

        _json['sequence'] = list()
        for s in self.sequence:
            _json['sequence'].append(s)

        return _json
