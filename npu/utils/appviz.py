# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from copy import copy, deepcopy
import json
import os
from typing import Tuple
import warnings
from IPython.display import display
import npu.utils.svg as svg
import npu.utils.svg_config as config

_ct_color = {
    0: {'kernel': config.dark_pink,
        'inbuf': [config.light_pink, config.light_red],
        'outbuf': [config.dark_pink, config.lilac]},
    1: {'kernel': config.yellow,
        'inbuf': [config.orange, config.purple],
        'outbuf': [config.yellow, config.orange]},
    2: {'kernel': config.dark_blue,
        'inbuf': [config.light_blue, config.lilac],
        'outbuf': [config.dark_blue, config.pink]},
    3: {'kernel': config.red,
        'inbuf': [config.light_red, config.green],
        'outbuf': [config.red, config.light_orange]}
}


class AppViz:
    """ Visualize the Dataflow Graph in a column of the NPU"""
    def __init__(self, metadata: str):
        self._kanimate_duration = 4
        self._md = self._parse_metadata(metadata)
        self._loc_conv = {2: 3, 3: 2, 4: 1, 5: 0}
        self._appname = self._md['application']
        self._col_svg = svg.RyzenAiColumn()
        self._ct_color = deepcopy(_ct_color)
        self._drawn_kernels = self._draw_kernels()
        self._ct2mt_counter = 0
        self._mt2ct_counter = 0
        self._mt2ct_passthrough = {'found': False, 'color': None}
        self._dbuf_colors = {}
        self._draw_connections_sorted()
        self._draw_key()

    def _draw_key(self) -> None:
        kernels = []
        for kname, k in self._drawn_kernels.items():
            p = tuple([kname, k['kcolor']])
            kernels.append(p)
        self._col_svg.generate_key(kernels)

    def _draw_kernels(self) -> dict:
        """ Draws the kernels onto the appropriate tiles
            Returns a dict containing a reference to all
            the drawn kernels that can be used for
            appending buffers and connections to the
            kernels"""
        drawn_kernels = {}
        for k in self._md['kernels'].values():
            if k['type'] == "CT":
                info = {}
                info['row'] = self._loc_conv[k['tloc'][1]]
                info['kcolor'] = self._ct_color[info['row']]['kernel']
                self._col_svg.aie_tiles[info['row']].add_kernel(
                        info['kcolor'],
                        self._kanimate_duration)
                drawn_kernels[k['name']] = info

        return drawn_kernels

    def _get_output_buffer(self, kernelidx: int):
        color = None
        if self._ct_color[kernelidx]['outbuf']:
            color = self._ct_color[kernelidx]['outbuf'][0]
            self._ct_color[kernelidx]['outbuf'].remove(color)
        if color is None:
            warnings.warn("Cannot display more than two output buffers for "
                          f"compute tile {kernelidx}")
        return color

    def _get_input_buffer(self, kernelidx: int):
        color = None
        if self._ct_color[kernelidx]['inbuf']:
            color = self._ct_color[kernelidx]['inbuf'][0]
            self._ct_color[kernelidx]['inbuf'].remove(color)
        if color is None:
            warnings.warn("Cannot display more than two input buffers for "
                          f"compute tile {kernelidx}")
        return color

    def _is_rtp_con(self, connection: dict) -> bool:
        """ determines if a connection is an RTP from the json """
        return connection['srckernel'] == 'user'

    def _draw_connections_sorted(self) -> None:
        """Draw connections with a pre-defined priority"""

        conn = copy(self._md['connections'])
        tmpconn = copy(self._md['connections'])

        # Draw animations starting in the IT first
        for k, c in conn.items():
            if not self._is_rtp_con(c):
                if c['srcport'] == 'ITout':
                    self._draw_connection(c)
                    tmpconn.pop(k)
            else:
                tmpconn.pop(k)
        conn = copy(tmpconn)

        # Draw animations starting in the MT second. Run twice for ping-pong
        for i in range(2):
            for k, c in conn.items():
                if c['srcport'] == 'MTout':
                    self._draw_connection(c, bool(i))
                    if i == 1:
                        tmpconn.pop(k)
            self._ct2mt_counter = 0
            self._mt2ct_counter = 0
        conn = copy(tmpconn)

        # Draw animations ending in the MT third. Run twice for ping-pong
        for i in range(2):
            for k, c in conn.items():
                if c['sinkport'] == 'MTin':
                    self._draw_connection(c, bool(i))
                    if i == 1:
                        tmpconn.pop(k)
        conn = copy(tmpconn)

        # Draw remaining animations
        for c in conn.values():
            self._draw_connection(c)

    def _draw_connection(self, c, dbuf: bool = False) -> None:
        """ Draws kernels, buffers and data movement

        Iterates through the connections, drawing the kernel, buffers and
        data movement.
        """
        src = self._md['kernels'][c['srckernel']]
        dst = self._md['kernels'][c['sinkkernel']]

        if src['type'] == 'CT' and dst['type'] == 'CT':
            src_row = self._drawn_kernels[src['name']]['row']
            dst_row = self._drawn_kernels[dst['name']]['row']
            for i in range(2):
                self._col_svg.aie_tiles[src_row].add_buffer(
                            self._drawn_kernels[src['name']]['kcolor'],
                            self._kanimate_duration/2,
                            start_empty=not bool(i))
                # if CTs are non neighbors we need to add double buffer in dst
                if not self._are_neighbors(src, dst):
                    self._col_svg.aie_tiles[dst_row].add_buffer(
                            self._drawn_kernels[src['name']]['kcolor'],
                            self._kanimate_duration/2,
                            start_empty= bool(i))

            self._draw_ct2ct_data_movement(src, dst)

        if src['type'] == 'IT' and dst['type'] == 'CT':
            dst_row = self._drawn_kernels[dst['name']]['row']
            bufcol = self._get_input_buffer(dst_row)
            for i in range(2):
                self._col_svg.aie_tiles[dst_row].add_buffer(
                            bufcol,
                            self._kanimate_duration/2,
                            start_empty=bool(i))
            self._draw_ub_ic_ingress(dst, bufcol)

        if src['type'] == 'CT' and dst['type'] == 'IT':
            src_row = self._drawn_kernels[src['name']]['row']
            bufcol = self._get_output_buffer(src_row)
            for i in range(2):
                self._col_svg.aie_tiles[src_row].add_buffer(
                            bufcol,
                            self._kanimate_duration/2,
                            start_empty=not bool(i))
            self._draw_ub_ic_egress(src)

        if src['type'] == 'CT' and dst['type'] == 'MT':
            src_row = self._drawn_kernels[src['name']]['row']
            if not dbuf:
                bufcol = self._get_output_buffer(src_row)
                self._dbuf_colors[c['name']] = bufcol
            else:
                bufcol = self._dbuf_colors[c['name']]

            self._col_svg.mem_tiles[0].add_buffer(
                        bufcol,
                        self._kanimate_duration/2,
                        start_empty=dbuf,
                        color2=config.purple,
                        delay= self._ct2mt_counter/5)
            if not dbuf:
                for i in range(2):
                    self._col_svg.aie_tiles[src_row].add_buffer(
                                bufcol,
                                self._kanimate_duration/2,
                                start_empty=bool(i))

                self._draw_ct2mem_ic(src)
            else:
                self._ct2mt_counter += 1

        if src['type'] == 'MT' and dst['type'] == 'CT':
            dst_row = self._drawn_kernels[dst['name']]['row']
            if not dbuf:
                dst_buf_color = self._get_input_buffer(dst_row)
                self._dbuf_colors[c['name']] = dst_buf_color
            else:
                dst_buf_color = self._dbuf_colors[c['name']]

            show_mem_buffer = True
            mtmode = src.get('mtmode')
            if mtmode == 'passthrough':
                if self._mt2ct_passthrough['found']:
                    dst_buf_color = self._mt2ct_passthrough['color']
                    show_mem_buffer = False
                else:
                    self._mt2ct_passthrough['found'] = True
                    self._mt2ct_passthrough['color'] = dst_buf_color

            if show_mem_buffer:
                for i in range(int(self._mt2ct_passthrough['found']) + 1):
                    self._col_svg.mem_tiles[0].add_buffer(
                                config.green,
                                self._kanimate_duration/2,
                                start_empty=dbuf ^ bool(i),
                                color2=dst_buf_color,
                                delay=self._mt2ct_counter/5)
            self._col_svg.aie_tiles[dst_row].add_buffer(
                        dst_buf_color,
                        self._kanimate_duration/2,
                        start_empty=dbuf)
            if not dbuf:
                self._draw_mem2ct_ic(dst, c, dst_buf_color, mtmode)
            else:
                self._mt2ct_counter += 1

        self._draw_ub2mem_ic(src, dst)
        self._draw_mem2ub_ic(src, dst)

    def _draw_ct2mem_ic(self, src) -> None:
        """Display animation originating from CT and destination MT"""

        src_color = self._drawn_kernels[src['name']]['kcolor']
        src_row = self._loc_conv[src['tloc'][1]]
        delay = self._ct2mt_counter / 5

        for i in range(3, src_row-1, -1):
            diagonal_from_tile = i == src_row
            self._col_svg.aie_tiles[i].add_ic_animation(
                    diagonal_from_tile=diagonal_from_tile,
                    south=1,
                    duration=self._kanimate_duration/2,
                    delay=delay,
                    color=src_color)

        self._col_svg.mem_tiles[0].add_ic_animation(
                    diagonal_to_tile=1,
                    duration=self._kanimate_duration/2,
                    delay=delay,
                    color=src_color)
        self._ct2mt_counter += 1

    def _draw_mem2ct_ic(self, dst, c, dst_color, mtmode=None) -> None:
        """Display animation originating from MT and destination CT"""

        dst_row = self._loc_conv[dst['tloc'][1]]
        delay = self._mt2ct_counter / 5

        self._col_svg.mem_tiles[0].add_ic_animation(
                    diagonal_from_tile=1,
                    duration=self._kanimate_duration/2,
                    delay=delay,
                    color=dst_color)

        for i in range(3, dst_row-1, -1):
            diagonal_to_tile = i == dst_row
            self._col_svg.aie_tiles[i].add_ic_animation(
                    diagonal_to_tile=diagonal_to_tile,
                    north=1,
                    duration=self._kanimate_duration/2,
                    delay=delay,
                    color=dst_color)
        self._mt2ct_counter += int(mtmode == 'split')

    def _draw_ub2mem_ic(self, src, dst) -> None:
        """Display animation originating from IT and destination MT"""

        if src['type'] == 'IT' and dst['type'] == 'MT':
            src_color = config.green
            self._col_svg.mem_tiles[0].add_ic_animation(
                        diagonal_to_tile=1,
                        north=1,
                        duration=self._kanimate_duration*1,
                        color=src_color)
            self._col_svg.if_tiles[0].add_dma_animation(
                        south_up=1,
                        duration=self._kanimate_duration/2,
                        color=src_color)
            self._col_svg.if_tiles[0].add_ic_animation(
                        diagonal_from_tile=1,
                        duration=self._kanimate_duration/2,
                        color=src_color)

    def _draw_mem2ub_ic(self, src, dst) -> None:
        """Display animation originating from MT and destination IT"""

        if src['type'] == 'MT' and dst['type'] == 'IT':
            dst_color = config.purple
            self._col_svg.mem_tiles[0].add_ic_animation(
                        diagonal_from_tile=1,
                        south=1,
                        duration=self._kanimate_duration*1,
                        color=dst_color)
            self._col_svg.if_tiles[0].add_ic_animation(
                        diagonal_to_tile=1,
                        duration=self._kanimate_duration/2,
                        color=dst_color)
            self._col_svg.if_tiles[0].add_dma_animation(
                        south_down=1,
                        duration=self._kanimate_duration/2,
                        color=dst_color)

    def _draw_ub_ic_egress(self, src) -> None:
        """Display animation originating from CT and destination IT"""

        src_color = self._drawn_kernels[src['name']]['kcolor']
        src_row = self._loc_conv[src['tloc'][1]]

        for i in range(3, src_row-1, -1):
            diagonal_from_tile = i == src_row
            self._col_svg.aie_tiles[i].add_ic_animation(
                    diagonal_from_tile=diagonal_from_tile,
                    south=1,
                    duration=self._kanimate_duration/2,
                    color=src_color)

        self._col_svg.mem_tiles[0].add_ic_animation(
                    south=1,
                    duration=self._kanimate_duration/2,
                    color=src_color)

        self._col_svg.if_tiles[0].add_ic_animation(
                    diagonal_to_tile=1,
                    duration=self._kanimate_duration/2,
                    color=src_color)

        self._col_svg.if_tiles[0].add_dma_animation(
                    south_down=1,
                    duration=self._kanimate_duration/2,
                    color=src_color)

    def _draw_ub_ic_ingress(self, dst, kcolor) -> None:
        """Display animation originating from IT and destination CT"""

        self._col_svg.if_tiles[0].add_ic_animation(
                    diagonal_from_tile=1,
                    duration=self._kanimate_duration/2,
                    color=kcolor)
        self._col_svg.if_tiles[0].add_dma_animation(
                    south_up=1,
                    duration=self._kanimate_duration/2,
                    color=kcolor)

        self._col_svg.mem_tiles[0].add_ic_animation(
                    north=1,
                    duration=self._kanimate_duration/2,
                    color=kcolor)

        dst_row = self._loc_conv[dst['tloc'][1]]
        for i in range(3, dst_row-1, -1):
            diagonal_to_tile = dst_row == i
            self._col_svg.aie_tiles[i].add_ic_animation(
                    diagonal_to_tile=diagonal_to_tile,
                    north=1,
                    duration=self._kanimate_duration/2,
                    color=kcolor)

    def _draw_ct2ct_data_movement(self, src, dst) -> None:
        """Display animation originating from CT and destination CT

        If the tiles are neighbor we display the animation using the
        crossbar.
        If they are not neighbors, we use the stream interconnect.
        """

        src_row = self._drawn_kernels[src['name']]['row']
        src_kcol = self._drawn_kernels[src['name']]['kcolor']
        if self._are_neighbors(src, dst):
            up, down = self._get_rel_neighbor_loc(src, dst)
            self._col_svg.aie_tiles[src_row-int(up)].add_mem_animation(
                    up_mem_to_aie=up,
                    down_mem_to_aie=down,
                    color=src_kcol,
                    duration=self._kanimate_duration)
        else:
            dst_row = self._drawn_kernels[dst['name']]['row']
            if src_row > dst_row:
                for i in range(dst_row, src_row+1):
                    diagonal_to_tile = dst_row == i
                    diagonal_from_tile = src_row == i
                    self._col_svg.aie_tiles[i].add_ic_animation(
                        diagonal_from_tile=diagonal_from_tile,
                        diagonal_to_tile=diagonal_to_tile,
                        north=not diagonal_from_tile,
                        duration=self._kanimate_duration/2,
                        color=src_kcol)
            else:
                for i in range(src_row, dst_row+1):
                    diagonal_to_tile = dst_row == i
                    diagonal_from_tile = src_row == i
                    self._col_svg.aie_tiles[i].add_ic_animation(
                        diagonal_from_tile=diagonal_from_tile,
                        diagonal_to_tile=diagonal_to_tile,
                        south=not diagonal_to_tile,
                        duration=self._kanimate_duration/2,
                        color=src_kcol)

    def _are_neighbors(self, aie1, aie2) -> str:
        """ Return true if two aie tiles are neighbors """
        row_diff = abs(aie1['tloc'][1] - aie2['tloc'][1])
        return row_diff <= 1

    def _get_rel_neighbor_loc(self, aie1, aie2) -> Tuple[int, int]:
        if self._are_neighbors(aie1, aie2):
            row_diff = aie1['tloc'][1] - aie2['tloc'][1]
            return row_diff == -1, row_diff == +1

    def _parse_metadata(self, metadata) -> dict:
        """ parses and checks the incoming metadata """
        # metadata is either a filepath or a direct string of the json
        if isinstance(metadata, dict):
            return metadata
        if os.path.isfile(metadata):
            with open(metadata, "r") as fp:
                md = json.load(fp)
        else:
            try:
                md = json.load(metadata)
            except Exception as e:
                raise RuntimeError("Unable to parse input metadata for "
                                   "visualizing as either a "
                                   "file or string") from e
        return md

    def save(self, filename: str = None) -> None:
        """saves animation to a file"""
        if filename is None:
            name = self._appname + '.svg'
        else:
            name = filename
        self._col_svg.generate_image(filename=name)

    @property
    def show(self) -> None:
        import tempfile
        from pathlib import Path
        tmp_dir = tempfile.mkdtemp()
        _t = Path(tmp_dir) / f"_{self._appname}_viz.svg"
        self._col_svg.generate_image(filename=_t)
        from IPython.core.display import SVG
        display(SVG(filename=_t))
