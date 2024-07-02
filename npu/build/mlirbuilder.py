# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .mlirtiles import CTTile, MEMTile, ITTile
from .buffers import MTBuffer
from .mlirconnections import MLIRConnect, ObjectFIFO
from .mlirsequencebuilder import MLIRSequnceBuilder 
from typing import Dict, List, Tuple
from collections import OrderedDict
from itertools import groupby
import ctypes


class MLIRBuilder:
    """This class builds an MLIR representation of an application starting from AppMetadata.

    Attributes
    ----------
    metadata : AppMetadata
        The application metadata.  
    app : JSON
        The json representation of an application metadata.
    kernels : dict
        Dictionary storing all unique compute tile kernels in this application.
    connections : dict
        Dictionary storing all unique connections between kernels in this application.
    sequence : list
        List of the application data movements between kernels.
    config : tuple
        Tuple containing number of CT, MT, IT tiles that can be used to build the MLIR.
    """

    def __init__(self, metadata, config=(4,1,1)):
        """Return a new MLIRBuilder object.""" 
        app = metadata.to_json()
        self.kernels = app['kernels']
        self.connections = app['connections']
        self.sequence = app['sequence']
        self.app = app
        self.config = config
        self.metadata = metadata

        MLIRConnect.reset_id()

        self.aietiles, self.memtiles, self.sdmatiles = self._parse_tiles(config)
        self.tiles = {**self.sdmatiles,  **self.memtiles, **self.aietiles} 
        self._map_kernels_to_tiles()

        self._cons_src2dst = self._populate_src2dst_cons_dict()
        self._cons_dst2src = self._populate_dst2src_cons_dict()
        self._cons_broadcasts = self._populate_broadcast_dict()

        self.objfifos = self._map_connections_to_objectfifos()
        self._map_objectfifos_to_tiles()
        self._validate_app()

        self.seqbuilder = MLIRSequnceBuilder(self.app, self.aietiles, self._cons_broadcasts)

    def _parse_tiles(self, config):
        """Return the Tile (x,y) locations based on the number of Tiles available."""
        aies  = {(ix // 4, ix % 4 + 2) : CTTile(ix // 4, ix % 4 + 2) for ix in range(config[0])}
        mts   = {(ix, 1) : MEMTile(ix, 1) for ix in range(config[1])}
        sdmas = {(ix, 0) : ITTile(ix, 0) for ix in range(config[1])}
        return aies, mts, sdmas


    def to_mlir(self, file=None):
        """Toplevel method to generate the application MLIR."""
        indent = "   "
        used_tiles = [tile for _, tile in self.tiles.items() if tile.is_used()]
        used_aie_tiles = [tile for _, tile in self.aietiles.items() if tile.is_used()]        

        s = 'module  {\n'
        s += f'{indent}AIE.device(ipu)'
        s +='{\n\n'

        for t in used_tiles:
            s += f'{t.to_mlir_tile_declare(indent)}\n'

        for o in self.objfifos:
            s += f'{indent}{o.to_mlir_declare()}\n'
        s += '\n'

        s += f'{self._link_objectfifos_via_memtile(indent)}\n'

        kds = list()
        for aie in used_aie_tiles:
            kds.append(f'{indent}{aie.to_mlir_kernel_declare()}')
        s += '\n'.join(set(kds))
        s += '\n\n'

        for _, aie in self.aietiles.items():
            s += aie.to_mlir()
            
        s += f'{self.seqbuilder.mlir}'
        s += ' }\n'
        s += "}\n"

        if file is None:
            return s

        with open(file, "w") as f:
            f.write(s)

        return ""        


    def _map_kernels_to_tiles(self):

        # First, preplace constrained AIE Tiles
        for _, k in self.kernels.items():
            if k['type'] == 'CT' and k['tloc'] is not None:

                if self.aietiles[k['tloc']].kernel is not None:
                    raise ValueError(f'Cannot place {k["name"]} kernel - CT tile previously constrained with {self.aietiles[k["tloc"]].kernel["name"]}')
                self.aietiles[k['tloc']].kernel = k  

        # Then AIE kernels onto any free AIE Tiles
        # ...place all buffers on first MT / SDMA tiles
        for kname, k in self.kernels.items():
            if k['type'] == 'CT':
                if k['tloc'] is None:
                    raise ValueError(f'{kname} needs constained to a tile location (.tloc) before builing.')
                pass

            elif k['type'] == 'MT':
                for _, mt in self.memtiles.items():
                    k['tloc'] = mt.tloc
                    mt.buffers[kname] = k
                    break 
            elif k['type'] == 'IT':
                for _, sdma in self.sdmatiles.items():
                    k['tloc'] = sdma.tloc
                    sdma.buffers[kname] = k
                    break
            else:
                raise ValueError(f'Cannot map kernel of type {k["type"]} to Array config {self.config}')

    def _populate_src2dst_cons_dict(self) -> Dict[Tuple[str,str], List[Tuple[str,str]]]:
        """ Creates a mapping dict of the connections where the key is the src (kernel, port) tuple and
        the value is a list of (kernel,port) tuple destinations."""
        cons_dict = {}
        for con in self.connections.values():
            con_src = (con['srckernel'], con['srcport'])
            con_dst = (con['sinkkernel'], con['sinkport'])
            if con_src in cons_dict:
                cons_dict[con_src].append(con_dst)
            else:
                cons_dict[con_src] = [con_dst]
        return cons_dict
    
    def _populate_dst2src_cons_dict(self) -> Dict[Tuple[str,str], List[Tuple[str,str]]]:
        """ Creates a mapping dict of the connections where the key is the dst (kernel, port) tuple and
        the value is a list of source (kernel,port) tuples."""
        cons_dict = {}
        for con in self.connections.values():
            con_src = (con['srckernel'], con['srcport'])
            con_dst = (con['sinkkernel'], con['sinkport'])
            if con_dst in cons_dict:
                cons_dict[con_dst].append(con_src)
            else:
                cons_dict[con_dst] = [con_src]
        return cons_dict  
    
    def _populate_broadcast_dict(self) -> Dict[Tuple[str,str], List[Tuple[str,str]]]:
        """ Filters the dict of src2dst connection mappings produced by _populate_src2dst_cons_dict
        to produce a dict that only contains the broadcast pattern where the same data is going
        to multiple destinations."""
        cons_src2dst = self._populate_src2dst_cons_dict()
        broadcasts = {}
        for src, dsts in cons_src2dst.items():
            if len(dsts) > 1:
                if src != ('user', 'write'): # skip over RTPs
                    bc_analysis = self._analyse_broadcast(src)
                    if bc_analysis == "BCAST":
                        broadcasts[src] = dsts
                    elif bc_analysis == "MIX":
                        raise RuntimeError(f"""
                         Mixing broadcasts and distributes from the same source is not yet supported 
                         {src=} to {dsts=}
                         """)
        return broadcasts

    def _all_equal(self,outgoing_shapes)->bool:
        reference_list = next(iter(outgoing_shapes.values()))
        for lst in outgoing_shapes.values():
            if lst != reference_list:
                return False
        return True


    def _all_unique(self, outgoing_shapes)->bool: 
        seen_list = []
        for lst in outgoing_shapes.values():
            if lst in seen_list:
                return False
        return True

    def _analyse_broadcast(self, src:Tuple[str,str]) -> str:
        """When given a source (kernel, port) tuple determines if it is a: 
        true broadcast, i.e. same data unique destinations (returns "BCAST");
        distribute op, i.e. different chunks of the data to different destinations (returns "DIST");
        a mix of both (returns "MIX").
        """
        outgoing_shapes = {} 
        for s in self.sequence:
            con_src = (s['srckernelname'], s['srcportname'])
            if con_src  == src:
                dst = (s['snkkernelname'], s['snkportname'])
                outgoing_shape = (s['srcslices'], s['srcoffset'], s['srcnbytes'])
                if dst not in outgoing_shapes:
                    outgoing_shapes[dst] = [] 
                outgoing_shapes[dst].append(outgoing_shape)
        if self._all_equal(outgoing_shapes):
            return "BCAST"
        elif self._all_unique(outgoing_shapes): 
            return "DIST"
        else:
            return "MIX"
                

    def _get_bcast_nbytes_offset(self, bcast_src:Tuple[str,str])->Tuple[int,int]:
        for s in self.sequence:
            cons_src = (s['srckernelname'], s['srcportname'])
            if bcast_src == cons_src:
                return s['srcnbytes'], s['srcoffset']
        raise RuntimeError("Unable to find any transfers for broadcast connection {bcast_src} in the sequence")

    def _map_connections_to_objectfifos(self):
        obfs = list()
        for s in [s for s in self.sequence if s['seqtype'] == 'buffer']:
            for c in [c for c in self.connections if s['name'] == c]:                
                if c in [obf.name for obf in obfs]:
                    # TODO : validate that channel transfer nbytes is consistent
                    break
                self.connections[c]['ctype'] = "objfifo,pingpong"                                      

                src = (self.connections[c]['srckernel'],
                       self.connections[c]['srcport'])

                if src not in self._cons_broadcasts:
                    dsts = [(self.connections[c]['sinkkernel'],
                             self.connections[c]['sinkport'])]

                    obfs.append(ObjectFIFO(c, src, dsts, s['nbytes'], s['offset'], self.tiles, self.kernels))
        
        # map broadcast connections
        for src, dsts in self._cons_broadcasts.items():
            nbytes, offset = self._get_bcast_nbytes_offset(src)
            obfs.append(ObjectFIFO(f"{src[0]}__{src[1]}", src, dsts, nbytes, offset, self.tiles, self.kernels)) 

        return obfs

    def _map_objectfifos_to_tiles(self):
        for obf in self.objfifos:
            for _, aie in self.aietiles.items():
                if aie.kernel is None:
                    continue
                if aie.kernel['name'] == obf.src[0]:
                    aie.objfifos_produce.append(obf)
                for d in obf.dsts:
                    if aie.kernel['name'] == d[0]:
                        aie.objfifos_consume.append(obf)

    def _get_mtbuffer_io(self)->Dict:
        cons = OrderedDict()
        for c in self.metadata.connections:
            cons[c.name] = c

        mt_buff_links = {}
        for c in cons.values():
            if isinstance(c.srckernel, MTBuffer):
                if c.srckernel.name not in mt_buff_links:
                    mt_buff_links[c.srckernel.name] = { 'in' : [], 'out' : [] }
                mt_buff_links[c.srckernel.name]['out'].append(c)
                mt_buff_links[c.srckernel.name]['out'] = self._sort_srcport_mtbuff_links(mt_buff_links[c.srckernel.name]['out'])
            elif isinstance(c.sinkkernel, MTBuffer):
                if c.sinkkernel.name not in mt_buff_links:
                    mt_buff_links[c.sinkkernel.name] = { 'in' : [], 'out' : [] }
                mt_buff_links[c.sinkkernel.name]['in'].append(c) 
                mt_buff_links[c.sinkkernel.name]['in'] = self._sort_sinkport_mtbuff_links(mt_buff_links[c.sinkkernel.name]['in'])
        return mt_buff_links

    def _sort_srcport_mtbuff_links(self, buff_links)->List:
        def sorting_key(connection):
            if not connection.srcport.slices:
                return float('inf')
            if len(connection.srcport.slices) == 0:
                return float('inf')
            if isinstance(connection.srcport.slices[0], int):
                return connection.srcport.slices[0]
            if isinstance(connection.srcport.slices[0], slice):
                return connection.srcport.slices[0].start
        return sorted(buff_links, key=sorting_key)
        
    def _sort_sinkport_mtbuff_links(self, buff_links)->List:
        def sorting_key(connection):
            if not connection.sinkport.slices:
                return float('inf')
            if len(connection.sinkport.slices) == 0:
                return float('inf')
            if isinstance(connection.sinkport.slices[0], int):
                return connection.sinkport.slices[0]
            if isinstance(connection.sinkport.slices[0], slice):
                return connection.sinkport.slices[0].start
        return sorted(buff_links, key=sorting_key)

    def _is_con_broadcast(self, connection)->bool:
        return (connection.srckernel.name, connection.srcport.name) in self._cons_broadcasts

    def _get_objectfifo_varname(self, connection)->str:
        if self._is_con_broadcast(connection):
            name = f"{connection.srckernel.name}__{connection.srcport.name}" 
        else:
            name = connection.name

        for obj in self.objfifos:
            if obj.name == name:
                return obj.varname

        raise RuntimeError(f"Unable to find an instantiation for objectFifo {name}")

    def _is_mtlink_bcast(self, link)->bool:
        for l in link['out']:
            if self._is_con_broadcast(l):
                return True
        return False

    def _link_objectfifos_via_memtile(self, indent='')->str:
        mt_buff_links = self._get_mtbuffer_io()
        s = ''
        for link in mt_buff_links.values():
            if not self._is_mtlink_bcast(link):
                s += self._generate_distribute_link(link, indent=indent)
            else:
                s += self._generate_broadcast_link(link, mt_buff_links, indent=indent) 

        return s

    def _generate_distribute_link(self, link, indent='')->str:
        s = f"{indent}AIE.objectFifo.link ["

        for i, src in enumerate(link['in']):
            s += f"{self._get_objectfifo_varname(src)}"
            if i != len(link['in']) - 1:
                s += ','
        s += ' ] -> ['

        for i, dst in enumerate(link['out']):
            s += f"{self._get_objectfifo_varname(dst)}"
            if i != len(link['out']) - 1:
                s += ','
        s += ']'
        s += ' ()\n'
        return s

    def _generate_broadcast_link(self, link, mtlinks, indent='')->str:
        if len(link['in']) > 1:
            raise RuntimeError(f"Trying to link a MT broadcast that is fed from multiple input src's this is not possibe {link=}")

        dst = (link['in'][0].srckernel.name, link['in'][0].srcport.name)

        for mtlink in mtlinks.values():
            if len(mtlink['in']) == 1:
                mtdst = (mtlink['in'][0].srckernel.name, mtlink['in'][0].srcport.name)
                if mtdst == dst:
                    s = f"{indent}AIE.objectFifo.link "
                    s += f"[{self._get_objectfifo_varname(mtlink['in'][0])} ] -> "
                    s += f"[{self._get_objectfifo_varname(link['out'][0])}] ()\n"
                    return s
        raise RuntimeError(f"Unable to find a feeding buffer to link to in the memtile for {link=}")


    def _validate_app(self):
        
        if len(self.kernels) == 0 or len(self.connections) == 0:
            raise ValueError(f'{len(self.kernels)} kernels or {len(self.connections)} connections cannot be zero')            

        if len(self.aietiles) > len(self.aietiles):
            raise ValueError(f'{len(self.kernels)} kernels cannot be placed onto {len(self.aietiles)} AIE tiles')
