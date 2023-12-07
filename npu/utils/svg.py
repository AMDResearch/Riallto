# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import warnings
import npu.utils.svg_config as config
import numpy as np

class Tile:
    def __init__(
        self,
        row=0,
        col=0,
        tile_type="",
        tile_background_color="#FFFFFF",
        container_background_color="#FFFFFF",
    ):
        self.x = config.get_tile_x(col)
        self.y = config.get_tile_y(row)
        self.index = col * config.ROWS + row
        self.tile_type = tile_type
        self.hide_tile_svg = ""

        self.tile_border = Box(
            self.tile_type,
            self.index,
            self.x,
            self.y,
            config.tile_width,
            config.tile_height,
            tile_background_color,
            stroke_width=2,
        )
        self.interconnect_boxes = Box(
            "interconnect_box",
            self.index,
            self.x + config.ic_offset,
            self.y + config.ic_offset,
            config.interconnect_box_width,
            config.interconnect_box_height,
            config.interconnect_color,
        )
        self.interconnect_icon = self._draw_interconnect_icon()
        self.container = Box(
            self.tile_type + "_container",
            self.index,
            self.x + config.aie_container_offset_x,
            self.y + config.aie_container_offset_y,
            config.aie_container_width,
            config.aie_container_height,
            container_background_color,
        )

        self.tile_svg = ""
        self.ic_connections_svg = ""
        self.ic_animations_svg = ""

        self.mem_animations_svg =""

        self.color = "#000000"
        self.draw_standalone_memory_connections = True
        self.draw_standalone_ic_connections = True
        self.draw_nsew_memory_connections = [0, 0, 0, 0]
        self.draw_nsew_ic_connections = [0, 0, 0, 0]
        self.memory_connections = ""

    def get_tile_svg(self):
        self.tile_svg = (
            self.tile_border.box
            + self.interconnect_boxes.box
            + self.interconnect_icon
            + self.container.box
        )
        return self.tile_svg

    def get_ic_connections_svg(self):
        return self.ic_connections_svg

    def get_ic_animations_svg(self):
        return self.ic_animations_svg

    def draw_ic_connections(self, north, south, east, west):
        line = self._draw_diagonal_ic()
        if north:
            x1 = self.x + config.ic_offset + int(config.interconnect_box_width / 2)
            y1 = self.y + config.ic_offset
            x2 = x1
            y2 = y1 - config.ic_offset - config.tile_gap
            line += f'<line id="v_ic_connection{self.index}" class="interconnect" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="red" stroke-width="1" fill="none" />\n'
        if south:
            x1 = self.x + config.ic_offset + int(config.interconnect_box_width / 2)
            y1 = self.y + config.ic_offset + config.interconnect_box_height
            x2 = x1
            y2 = y1 + config.ic_height
            line += f'<line id="v_ic_connection{self.index}" class="interconnect" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="red" stroke-width="1" fill="none" />\n'
        if east:
            x1 = self.x + config.ic_offset + config.interconnect_box_width
            y1 = self.y + config.ic_offset + int(config.interconnect_box_height / 2)
            x2 = x1 + config.ic_width
            y2 = y1
            line += f'<line id="h_ic_connection{self.index}" class="interconnect" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="red" stroke-width="1" fill="none" />\n'
        if west:
            x1 = self.x + config.ic_offset
            y1 = self.y + config.ic_offset + int(config.interconnect_box_height / 2)
            x2 = x1 - config.ic_offset - config.tile_gap
            y2 = y1
            line += f'<line id="h_ic_connection{self.index}" class="interconnect" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="red" stroke-width="1" fill="none" />\n'

        self.ic_connections_svg += line

    def _draw_interconnect_icon(self):
        length = config.interconnect_box_width / 2
        start = length / 2
        arrow_length = 4
        arrow_offset = arrow_length / 2
        i1 = self.x + config.ic_offset + start
        j1 = self.y + config.ic_offset + start
        i2 = i1 + length
        j2 = j1 + length
        h_line1 = f'<line id="ic_h_line{self.index}_0" class="interconnect" x1="{i1-arrow_offset}" y1="{j1+5}" x2="{i2-arrow_offset}" y2="{j1+5}" \
            stroke="black" stroke-width="1" fill="none" marker-end="url(#ic_endarrow)" />\n'
        h_line2 = f'<line id="ic_h_line{self.index}_1" class="interconnect" x1="{i2+arrow_offset}" y1="{j2-5}" x2="{i1+arrow_offset}" y2="{j2-5}" \
            stroke="black" stroke-width="1" fill="none" marker-end="url(#ic_endarrow)"/>\n'
        v_line1 = f'<line id="ic_h_line{self.index}_2" class="interconnect" x1="{i1+5}" y1="{j2+arrow_offset}" x2="{i1+5}" y2="{j1+arrow_offset}" \
            stroke="black" stroke-width="1" fill="none" marker-end="url(#ic_endarrow)"/>\n'
        v_line2 = f'<line id="ic_h_line{self.index}_3" class="interconnect" x1="{i2-5}" y1="{j1-arrow_offset}" x2="{i2-5}" y2="{j2-arrow_offset}" \
            stroke="black" stroke-width="1" fill="none" marker-end="url(#ic_endarrow)"/>\n'
        return h_line1 + h_line2 + v_line1 + v_line2

    def _draw_diagonal_ic(self):
        i1 = self.x + config.ic_offset + config.interconnect_box_width
        j1 = self.y + config.ic_offset + config.interconnect_box_width
        i2 = i1 + config.diagonal_offset
        j2 = j1 + config.diagonal_offset
        diagonal = f'<line id="diagonal{self.index}" class="interconnect" x1="{i1}" y1="{j1}" x2="{i2}" y2="{j2}" stroke="red" stroke-width="1" fill="none" />\n'
        return diagonal
        # Todo: add arrows:
        # marker-start="url(#red_startarrow)" marker-end="url(#red_endarrow)"

    def add_ic_animation(self, diagonal_to_tile=0, diagonal_from_tile=0, north=0, south=0, east=0, west=0, duration=2, delay=0, color="red"):

        # Diagonal line
        if diagonal_to_tile == 1:
            i1 = self.x + config.ic_offset + config.interconnect_box_width
            j1 = self.y + config.ic_offset + config.interconnect_box_width

            self.ic_animations_svg += f'<circle id="d_circle{self.index}" class="ic_animations" cx="{i1}" cy="{j1}" r="3" fill="none">\
            <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                path="M0, 0, \
                L{config.diagonal_offset}, {config.diagonal_offset},\
                "/>\
            <animate attributeName="fill"\
                values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                />\
            </circle>\n'
        # Diagonal line
        if diagonal_from_tile == 1:
            i1 = self.x + config.ic_offset + config.interconnect_box_width + config.diagonal_offset
            j1 = self.y + config.ic_offset + config.interconnect_box_height + config.diagonal_offset
            self.ic_animations_svg += f'<circle id="d_circle{self.index}" class="ic_animations" cx="{i1}" cy="{j1}" r="3" fill="none">\
            <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                path="M0, 0, \
                L{-config.diagonal_offset}, {-config.diagonal_offset},\
                "/>\
            <animate attributeName="fill"\
                values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                />\
            </circle>\n'

        if north == 1:
            x1 = self.x + config.ic_offset + int(config.interconnect_box_width / 2)
            y1 = self.y + config.ic_offset + config.interconnect_box_height+config.ic_height
            self.ic_animations_svg += f'<circle id="v_circle_north{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
            <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                path="M0, 0, \
                L0, {-config.ic_height}\
                " />\
            <animate attributeName="fill"\
                values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                />\
            </circle>\n'

        if south == 1:
            x1 = self.x + config.ic_offset + int(config.interconnect_box_width / 2)
            y1 = self.y + config.ic_offset + config.interconnect_box_height

            self.ic_animations_svg += f'<circle id="v_circle_south{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
            <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                path="M0, 0, \
                L0, {config.ic_height}\
                " />\
            <animate attributeName="fill"\
                values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                />\
            </circle>\n'

        if east == 1:
            x1 = self.x + config.ic_offset + config.interconnect_box_width
            y1 = self.y + config.ic_offset + int(config.interconnect_box_height / 2)

            self.ic_animations_svg += f'<circle id="h_circle_east{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                        path="M0, 0, \
                        L{config.ic_width}, 0\
                        " />\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

        if west == 1:
            x1 = self.x + config.ic_offset + config.interconnect_box_width+config.ic_width
            y1 = self.y + config.ic_offset + int(config.interconnect_box_height / 2)

            self.ic_animations_svg += f'<circle id="h_circle_west{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                        path="M0, 0, \
                        L{-config.ic_width}, 0\
                        " />\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

    def add_single_tile_ic_animation(self, north_up=0, north_down=0, west_left=0, west_right=0, duration=2, delay=0, color="red"):
        if north_up == 1:
            x1 = self.x + config.ic_offset + int(config.interconnect_box_width / 2)
            y1 = self.y + config.ic_offset
            self.ic_animations_svg += f'<circle id="v_circle_north{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
            <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                path="M0, 0, \
                L0, {-config.ic_offset-config.tile_gap}\
                " />\
            <animate attributeName="fill"\
                values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                />\
            </circle>\n'

        if north_down == 1:
            x1 = self.x + config.ic_offset + int(config.interconnect_box_width / 2)
            y1 = self.y + config.ic_offset -config.ic_offset-config.tile_gap

            self.ic_animations_svg += f'<circle id="v_circle_south{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
            <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                path="M0, 0, \
                L0, {config.ic_offset+config.tile_gap}\
                " />\
            <animate attributeName="fill"\
                values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                />\
            </circle>\n'

        if west_left == 1:
            x1 = self.x + config.ic_offset
            y1 = self.y + config.ic_offset + int(config.interconnect_box_height / 2)

            self.ic_animations_svg += f'<circle id="h_circle_east{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                        path="M0, 0, \
                        L{-config.ic_offset-config.tile_gap}, 0\
                        " />\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

        if west_right == 1:
            x1 = self.x - config.tile_gap
            y1 = self.y + config.ic_offset + int(config.interconnect_box_height / 2)

            self.ic_animations_svg += f'<circle id="h_circle_west{self.index}" class="ic_animations" cx="{x1}" cy="{y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite" begin="{delay}s"\
                        path="M0, 0, \
                        L{config.ic_offset+config.tile_gap}, 0\
                        " />\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

    def add_single_tile_mem_animation(self, north_up=0, north_down=0,
                                      south_up=0, south_down=0, east_left=0,
                                      east_right=0, west_right=0, west_left=0,
                                      color="blue", duration=2, delay=0):

        if north_up == 1:
            x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
            y1 = self.y+config.aie_container_offset_y
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l 0, {-config.aie_container_offset_y - config.tile_gap}\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if north_down == 1:
            x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
            y1 = self.y-config.tile_gap
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l 0, {config.aie_container_offset_y +config.tile_gap}\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if south_up == 1:
            x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
            y1 = self.y+config.tile_height+config.tile_gap
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l 0, {-config.ic_offset- config.tile_gap}\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if south_down == 1:
            x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
            y1 = self.y + config.tile_height - config.ic_offset
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l 0, {config.ic_offset +config.tile_gap}\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

        if east_left == 1:
            x1 = self.x+config.tile_width+config.tile_gap
            y1 = self.y+config.aie_box_offset_y+config.aie_box_height/2
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l{-config.ic_offset-config.tile_gap-config.container_offset}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if east_right == 1:
            x1 = self.x + config.tile_width- config.ic_offset-config.container_offset
            y1 = self.y+config.aie_box_offset_y+config.aie_box_height/2
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l{config.container_offset+config.ic_offset+config.tile_gap}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

        if west_left == 1:
            x1 = self.x + config.aie_box_offset_x
            y1 = self.y + config.aie_box_offset_y+config.aie_box_height/2
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l{-config.aie_box_offset_x-config.tile_gap}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if west_right == 1:
            x1 = self.x - config.tile_gap
            y1 = self.y+config.aie_box_offset_y+config.aie_box_height/2
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l{config.aie_box_offset_x+config.tile_gap}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

    def clear_ic_animation(self):
        self.ic_animations_svg = ""

    def hide_tile(self, opacity=0.5):
        self.hide_tile_svg = f'<rect id="opaque_box_{self.index}" class="opaque" width="{config.tile_width+config.tile_gap}" height="{config.tile_height+config.tile_gap}" \
            x="{self.x-config.tile_gap/2}" y="{self.y-config.tile_gap/2}" fill="{config.background}" opacity="{opacity}"/>\n'

    def show_tile(self):
        self.hide_tile_svg=""

    def get_hide_tile_svg(self):
        return self.hide_tile_svg

    def get_memory_connections_svg(self):
        return self.memory_connections

    def get_memory_animations_svg(self):
        return self.mem_animations_svg

    def show_all(self):
        self.hide_tile_svg = ""

class Image():
    def __init__(self):
        self.key_width = 0
        self.image_width = 0
        self.image_height = 0

    def _generate_svg_header(self, image_width, image_height, viewBox_height):
        return f'''
            <svg id="svg"
            version="1.1" 
            xmlns="http://www.w3.org/2000/svg" 
            xmlns:xlink="http://www.w3.org/1999/xlink" 
            width="{image_width}" 
            height="{image_height}" 
            viewBox="0, 0, {image_width}, {viewBox_height}">
            <defs>
                <marker id="ic_startarrow" markerWidth="10" markerHeight="7" 
                refX="10" refY="3.5" orient="auto">
                <polygon points="0 0, 4 2, 0 4" fill="black" />
                </marker>
                <marker id="ic_endarrow" markerWidth="7" markerHeight="4" 
                refX="0" refY="2" orient="auto" markerUnits="strokeWidth">
                    <polygon points="0 0, 4 2, 0 4" fill="black" />
                </marker>

                <marker id="red_startarrow" markerWidth="10" markerHeight="7" 
                refX="0" refY="2" orient="auto">
                <polygon points="0 2, 4 0, 4 4" fill="red" />
                </marker>
                <marker id="red_endarrow" markerWidth="7" markerHeight="4" 
                refX="4" refY="2" orient="auto" markerUnits="strokeWidth">
                    <polygon points="0 0, 4 2, 0 4" fill="red" />
                </marker>

                <marker id="startarrow" markerWidth="10" markerHeight="7" 
                refX="10" refY="3.5" orient="auto">
                <polygon points="0 0, 4 2, 0 4" fill="black" />
                </marker>
                <marker id="endarrow" markerWidth="10" markerHeight="7" 
                refX="0" refY="3.5" orient="auto" markerUnits="strokeWidth">
                    <polygon points="0 0, 4 2, 0 4" fill="black" />
                </marker>
            </defs>
            <style>\
                .ic_animations {{ \
                    opacity: {0.8}; \
                }}\
            </style>
            '''
    def _generate_svg_border(self, border_width, border_height, border_x, border_y, background):
        return f'<rect id="border" class="border" width="{border_width}" height="{border_height}" \
        x="{border_x}" y="{border_y}" stroke="black" stroke-width="1" fill="{background}" />\n'

    def generate_svg_file(self, svg_content, filename, show_border=1,
                          show_sysmem = False):
        OPEN_SVG = '<g id="svg">'
        CLOSE_SVG = '</g></svg>'
        # Check if rows and cols exist, and size image to include mem and if tiles

        if hasattr(self, 'cols') and hasattr(self, 'rows'):
            self.image_width = config.x_start * 2 + (config.tile_width + config.tile_gap) * self.cols - config.tile_gap + self.key_width
            self.image_height = (
                config.y_start * 2
                + (config.tile_height + config.tile_gap) * self.rows
                + (config.tile_height + config.tile_gap) * 2
                - config.tile_gap
            )
        else: # individual tile
            self.image_width = config.x_start * 2 + config.tile_width
            self.image_height = config.y_start * 2 + config.tile_height

        if show_border==1:
            border_x = config.x_start / 2
            border_y = config.y_start / 2

            border_width = self.image_width - config.x_start
            border_height = self.image_height - config.y_start
            svg_outer_border = self._generate_svg_border(border_width, border_height, border_x, border_y, config.background)
        else:
            svg_outer_border = ""

        if show_sysmem:
            viewBox_height = self.image_height + config.aie_box_height + 5
        else:
            viewBox_height = self.image_height

        svg_header = self._generate_svg_header(self.image_width, self.image_height, viewBox_height)
        svg_file = svg_header+OPEN_SVG+svg_outer_border+svg_content+CLOSE_SVG

        with open(filename, "w", newline='\n') as f:
            f.write(svg_file)

    def show_key(self):
        self.key_width = config.tile_width + config.tile_gap
        #self.key_svg = get_key()

    def hide_key(self):
        self.key_width = 0
        self.key_svg = ""

class Kernel():
    def __init__(self, x, y, index, color, duration, delay):
        self.index = index
        self.x0 = x + config.aie_box_offset_x + config.aie_box_width / 2
        self.y0 = y + config.aie_box_offset_y + config.aie_box_height / 2

        kernel = f'<linearGradient id="kernel_{self.index}" ' \
                 f'gradientTransform="rotate(90)"> ' \
                 f'<stop offset="0%" stop-color="white" /> ' \
                 f'<stop offset="10%" stop-color="{color}" /> ' \
                 f'</linearGradient> ' \
                 f'<circle id="kernel_circle_{self.index}" class="kernel" ' \
                 f'cx="{self.x0}" cy="{self.y0}" r="20" ' \
                 f'fill="url(#kernel_{self.index})" stroke="black" ' \
                 f'stroke-width="1" stroke-dasharray="5, 5"> ' \
                 f'<animateTransform attributeName="transform" type="rotate" '\
                 f'from="0 {self.x0} {self.y0}" to="360 {self.x0} {self.y0}" '\
                 f'dur="{duration}s" begin="{delay}s" '\
                 f'repeatCount="indefinite" /> </circle> \n'
        self.kernel_svg = kernel

    def get_kernel_svg(self):
        return self.kernel_svg

class KernelArc(Kernel):
    """Generate Sequential Kernels with Arcs"""
    def __init__(self, x:int, y:int, index:int, colors:list, duration:int,
                 radius:int=20, animate:bool=True):

        self.index = index
        self.radius = radius
        self.x0 = x + config.aie_box_offset_x + config.aie_box_width / 2
        self.y0 = y + config.aie_box_offset_y + config.aie_box_height / 2
        self.kernel_svg = ""
        self.num_kernels = len(colors)
        self.duration = duration / self.num_kernels
        arc = 360 / self.num_kernels

        if animate:
            for idx, k in enumerate(colors):
                start_angle = arc * idx
                end_angle = arc + arc * idx
                self.kernel_svg += self._get_arc(str(index)+str(idx), k,
                                                start_angle, end_angle, False)
                if animate:
                    for i in range(self.num_kernels):
                        start_angle_1 = arc * i
                        end_angle_1 = arc + arc * i
                        gidx = idx * self.num_kernels + i
                        if i > 0:
                            trigger = gidx - 1
                        else:
                            trigger = idx*self.num_kernels+self.num_kernels-1

                        self.kernel_svg += \
                            self._get_transform(gidx, start_angle_1,
                                                end_angle_1, i==0, trigger)
                        self.kernel_svg += '\n'

                trigger = self.num_kernels - idx - 1
                self.kernel_svg += \
                    self._get_animation(False, idx!=0,
                                        (trigger + 1) % self.num_kernels)
                self.kernel_svg += '\n'
                self.kernel_svg += self._get_animation(True, idx==0, trigger)
                self.kernel_svg += '\n'
                self.kernel_svg += '</path>\n'

    def _get_arc(self, index:str, color:str, start_angle:int, end_angle:int,
                 background:bool=False):
        id = f'id="kernel_circle_{index}"' if not background else ""
        bg = f'<path {id} ' \
             f'{str(self._describe_arc(start_angle, end_angle))} '\
             f'fill="{color}"'
        if not background:
            bg += ' stroke="black" stroke-width="1" stroke-dasharray="5, 5"'

        return bg + '>\n'

    def _get_animation(self, show:bool, zero_trigger:bool,
                       event_kernel:int):
        zero_trigger = "0s;" if zero_trigger else ""
        event = 'kernel_anim_' + str(self.index) + str(event_kernel) + '.end'
        animation = f'    <animate attributeName="stroke-opacity" ' \
                    f'from="{str(int(not show))}" to="{str(int(show))}" ' \
                    f'dur="0.0001s" fill="freeze" ' \
                    f'begin="{zero_trigger}{event}"' \
                    f'/>'
        return animation

    def _get_transform(self, idx:int, start_angle:int, end_angle:int,
                       zero_trigger:bool, event_kernel:int):
        """Generate angular rotation"""
        zero_trigger = "0s;" if zero_trigger else ""
        event = 'kernel_anim_' + str(self.index) + str(event_kernel) + '.end'

        kernel_idx = str(self.index) + str(idx)
        transform = f'    <animateTransform attributeName="transform" '\
                    f' type="rotate" id="kernel_anim_{kernel_idx}" ' \
                    f'from="{start_angle} {self.x0} {self.y0}" ' \
                    f'to="{end_angle} {self.x0} {self.y0}" '\
                    f'dur="{self.duration/self.num_kernels}s" ' \
                    f'begin="{zero_trigger}{event}" '\
                    'repeatCount="freeze"/> '
        return transform

    def _polar_to_cartesian(self, angle):
        """Compute cartesian coordinates"""
        # Offset the angle 90 degrees to start at 12 o'clock
        theta = (angle-90) * np.pi / 180.0
        point = self.radius * np.exp( 1j * theta )
        return self.x0 + point.real, self.y0 + point.imag

    def _describe_arc(self, start_angle, end_angle):
        """Generate arc"""

        start = self._polar_to_cartesian(end_angle)
        end = self._polar_to_cartesian(start_angle)

        arc_sweep = 0 if (start_angle - end_angle <= 180) else 1

        arc = f'd="M {start[0]} {start[1]} ' \
              f'A {self.radius} {self.radius} {0} {arc_sweep} {0} {end[0]} {end[1]} ' \
              f'L {self.x0} {self.y0} ' \
              f'L {start[0]} {start[1]}"'

        return arc

class AieTile(Tile, Image):

    MAX_BUFFERS = 8
    def __init__(self, col, row, label=False):
        super().__init__(col, row, "aie_tile", config.aie_tile_bg_color, config.aie_container_bg_color)
        self.aie_svg = ""
        self.number_of_buffers = 0
        self.buffers_svg = ""

        self.aie_box = Box(
            "aie_box",
            self.index,
            self.x + config.aie_box_offset_x,
            self.y + config.aie_box_offset_y,
            config.aie_box_width,
            config.aie_box_height,
            "white",
            "black",
            text1="AI",
            text2 = "Engine",
            font_size=config.font_size-1,
            text_color="black",
            show_label=label
        )
        self.local_memory_box = MemoryBox(
            "local_memory_box",
            self.index,
            self.x + config.local_memory_offset_x,
            self.y + config.local_memory_offset_y,
            config.local_memory_width,
            config.local_memory_height,
            "white",
            "black",
            text1="Data",
            text2="Memory",
            font_size=config.font_size-1,
            text_color="black",
            show_label=label,
            max_buffers = self.MAX_BUFFERS
        )

        self.aie_key_svg = ""
        self.aie_kernel_svg = ""
        kernels = []

        # Calculate local memory paths

        # Internal mem connection
        self.internal_mem_x1 = self.x + config.aie_box_offset_x + config.aie_box_width
        self.internal_mem_y1 = self.y + config.aie_box_offset_y + config.aie_box_height / 2
        self.internal_mem_x2 = config.local_memory_offset_x - (config.aie_box_offset_x + config.aie_box_width)
        self.internal_mem_y2 = self.internal_mem_y1

        self.mem_path_east_x1 = self.x + config.local_memory_offset_x + config.local_memory_width
        self.mem_path_east_y1 = self.y + config.aie_container_offset_y + int(config.aie_container_height / 2)
        self.mem_path_east_x2 = config.tile_width + config.tile_gap - config.aie_container_width+2*config.container_offset
        self.mem_path_east_y2 = 0

        self.mem_path_left_x1 = self.x + config.aie_box_offset_x +(config.tile_width+config.tile_gap)
        self.mem_path_left_y1 = self.mem_path_east_y1
        self.mem_path_left_x2 = -self.mem_path_east_x2
        self.mem_path_left_y2 = 0

        self.mem_path_west_x1 = self.x + config.aie_box_offset_x
        self.mem_path_west_y1 = self.mem_path_east_y1
        self.mem_path_west_x2 = -(self.mem_path_east_x2)
        self.mem_path_west_y2 = 0

        self.mem_path_north_x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
        self.mem_path_north_y1 = self.y+config.aie_container_offset_y
        #self.mem_path_north_x2 = 0
        self.mem_path_north_y2 = - config.tile_height - config.tile_gap + config.aie_container_height

        self.mem_path_south_x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
        self.mem_path_south_y1 = self.y+config.aie_container_offset_y+config.aie_container_height
        #self.mem_path_south_x2 = 0
        self.mem_path_south_y2 = config.tile_height+config.tile_gap -config.aie_container_height

        # Angled paths from aie/mem to aie/mem
        self.mem_path_south_mem_x1 = self.x + config.local_memory_offset_x + config.local_memory_width / 2
        self.mem_path_south_mem_y1 = self.y + config.local_memory_offset_y + config.local_memory_height

        self.mem_path_south_aie_x1 = self.x + config.aie_box_offset_x + config.aie_box_width / 2
        self.mem_path_south_aie_y1 = self.mem_path_south_mem_y1

        # Diagonal coords
        self.mem_path_south_diag_x2 = config.aie_container_offset_x + config.aie_container_width / 2 - (config.local_memory_offset_x + config.local_memory_width / 2)
        self.mem_path_south_diag_y2 = config.tile_height - (config.local_memory_offset_y + config.local_memory_height)

        # Straight line down
        self.mem_path_south_x3 = 0
        self.mem_path_south_y3 = (
            + config.tile_gap
            + config.aie_container_offset_y
            - config.tile_height
            + config.aie_container_offset_y
            + config.aie_container_height
        )
        # diagonal:
        self.mem_path_south_aie_x4 = config.aie_box_offset_x + config.aie_box_width/2 - (config.aie_container_offset_x + config.aie_container_width / 2)
        self.mem_path_south_aie_y4 = config.tile_height - (config.local_memory_offset_y + config.local_memory_height)

        self.mem_path_south_mem_x4 = config.local_memory_offset_x + config.local_memory_width / 2
        self.mem_path_south_mem_y4 = self.mem_path_south_aie_y4

    def add_memory_outlines(self):
         self.local_memory_box.add_memory_outlines(self.MAX_BUFFERS)

    def get_aie_svg(self):
        self.aie_svg = self.get_tile_svg() + self.aie_box.box + self.local_memory_box.box
        return self.aie_svg

    def get_aie_kernels_svg(self):
        return self.aie_kernel_svg

    def get_aie_buffers_svg(self):
        return self.buffers_svg

    #def get_key_svg(self):
    #    return self.aie_key_svg
        
    def draw_memory_connections(self, south_mem_to_aie=0, south_aie_to_mem=0, east=0, south=0, north=0, west=0, color="blue", strokewidth=1):
        line = ""
        tile_centered = 0
        self.south_paths = []
        if south_mem_to_aie:
            path = f'M {self.mem_path_south_mem_x1}, {self.mem_path_south_mem_y1}, \
                l{self.mem_path_south_x3}, {self.mem_path_south_y3},\
                l{self.mem_path_south_diag_x2}, {self.mem_path_south_diag_y2},\
                l{self.mem_path_south_aie_x4}, {self.mem_path_south_aie_y4}'
            line += f'<path id="south_mem_connection{self.index}" class="mem_connections" stroke="{color}" stroke-width="{strokewidth}" fill="none"\
                d="{path}"\
                />\n'
            self.path_down_mem_to_aie = path

            path = f'M {self.mem_path_south_aie_x1}, {self.mem_path_south_mem_y1+config.tile_height+config.tile_gap-config.aie_box_height}, \
                l{-self.mem_path_south_diag_x2}, {-self.mem_path_south_diag_y2},\
                l{-self.mem_path_south_diag_x2}, {-self.mem_path_south_diag_y2},\
                l{+self.mem_path_south_x3}, {-self.mem_path_south_y3}'

            self.path_up_aie_to_mem = path

        if south_aie_to_mem:
            path = f'M {self.mem_path_south_aie_x1}, {self.mem_path_south_aie_y1}, \
                l{self.mem_path_south_x3}, {self.mem_path_south_y3},\
                l{-self.mem_path_south_diag_x2}, {self.mem_path_south_diag_y2},\
                l{-self.mem_path_south_diag_x2}, {self.mem_path_south_diag_y2}'
            line += f'<path id="south_mem_connection{self.index}" class="mem_connections" stroke="{color}" stroke-width="{strokewidth}" fill="none"\
                d="{path}"\
                />\n'
            self.path_down_aie_to_mem = path

            path = f'M {self.mem_path_south_mem_x1},{self.mem_path_south_mem_y1+config.tile_height+config.tile_gap-config.aie_box_height}, \
                l{self.mem_path_south_diag_x2}, {-self.mem_path_south_diag_y2},\
                l{self.mem_path_south_diag_x2}, {-self.mem_path_south_diag_y2},\
                l{self.mem_path_south_x3}, {-self.mem_path_south_y3}'
            self.path_up_mem_to_aie = path
            
        if north:
            line += f'<path id="north_mem_connection{self.index}" class="mem_connections" stroke="{color}" stroke-width="{strokewidth}" fill="none"\
                d="M {self.mem_path_north_x1}, {self.mem_path_north_y1}, \
                l0, {self.mem_path_north_y2},\
                "\
                />\n'
        if south:
            path = f'M {self.mem_path_south_x1}, {self.mem_path_south_y1}, \
                l0, {self.mem_path_south_y2}'
            line += f'<path id="south_mem_connection{self.index}" class="mem_connections" stroke="{color}" stroke-width="{strokewidth}" fill="none"\
                d="{path}"\
                />\n'
            self.south_path = path
        if east:
            line += f'<line id="east_mem_connection{self.index}" x1="{self.mem_path_east_x1}" y1="{self.mem_path_east_y1}" \
                x2="{self.mem_path_east_x1+self.mem_path_east_x2}" y2="{self.mem_path_east_y1}" \
                class="mem_connections" stroke="{color}" stroke-width="{strokewidth}" fill="none" />\n'
        if west:
            line += f'<line id="west_mem_connection{self.index}" x1="{self.mem_path_west_x1}" y1="{self.mem_path_west_y1}" \
                x2="{self.mem_path_west_x1+self.mem_path_west_x2}" y2="{self.mem_path_west_y1}" \
                stroke="{color}" stroke-width="{strokewidth}" fill="none" />\n'

        # Internal memory connection
        line += f'<line id="internal_mem_connection{self.index}" x1="{self.internal_mem_x1}" y1="{self.internal_mem_y1}" \
                x2="{self.internal_mem_x1+ self.internal_mem_x2}" y2="{self.internal_mem_y2}" \
                class="mem_connections"  stroke="{color}" stroke-width="{strokewidth}" fill="none" />\n'

        self.memory_connections += line

    def add_kernel(self, color, duration=2, delay=0):
        self.aie_box.show_label = False
        kernel = Kernel(self.x,self.y, self.index, color, duration, delay)
        self.aie_kernel_svg = kernel.get_kernel_svg()

    def add_multiple_kernel(self, dict_kernels:list, animate:bool=True):
        """Dictionary should follow the format:
           {'duration': int, 'kernels': [color0, color1, ..., colorn]}
        """
        self.aie_box.show_label = False
        if not isinstance(dict_kernels, dict):
            raise TypeError("dict_kernels is not a dict")
        if len(dict_kernels['kernels']) < 1:
            raise TypeError("kernels has not enough elements")

        kernels = KernelArc(self.x, self.y, self.index,
                            dict_kernels['kernels'], dict_kernels['duration'],
                            20, animate)
        self.aie_kernel_svg += kernels.get_kernel_svg()

    def add_buffer(self, color, duration=2, start_empty=1, delay=0, color2=""):
        self.buffers_svg += \
            self.local_memory_box.add_buffer(color, duration, start_empty,
                                             delay, color2)

    def add_mem_animation(self, up_mem_to_aie=0, up_aie_to_mem=0, down_mem_to_aie=0, down_aie_to_mem=0, right=0, left=0, internal_left=0, internal_right=0, up=0, down=0, color="blue", duration=2, delay=0):
        if right == 1:
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="{self.mem_path_east_x1}" cy="{self.mem_path_east_y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M0, 0, \
                        L{self.mem_path_east_x2}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if left == 1:
            self.mem_animations_svg += f'<circle id="mem_circle_west{self.index}" class="mem_animations" cx="{self.mem_path_left_x1}" cy="{self.mem_path_left_y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M0, 0, \
                        L{self.mem_path_left_x2}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if up == 1: # North up
            x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
            y1 = self.y+config.aie_container_offset_y
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l 0, {-config.tile_height-config.tile_gap+config.aie_container_height }\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if down == 1: # North down
            x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
            y1 = self.y-config.tile_height-config.tile_gap+config.aie_container_offset_y+config.aie_container_height
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M {x1}, {y1}, \
                        l 0, {config.tile_height+config.tile_gap-config.aie_container_height}\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

        if up_mem_to_aie == 1: # need to fix coordinates; set fill to none for now.
            self.mem_animations_svg += f'<circle id="mem_circle_north{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="{self.path_up_mem_to_aie}" begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if up_aie_to_mem == 1:
            self.mem_animations_svg += f'<circle id="mem_circle_north{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="{self.path_up_aie_to_mem}" begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if down_mem_to_aie == 1: # south_down
            self.mem_animations_svg += f'<circle id="mem_circle_south{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="{self.path_down_mem_to_aie}" begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if down_aie_to_mem == 1:
            self.mem_animations_svg += f'<circle id="mem_circle_south{self.index}" class="mem_animations" cx="0" cy="0" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="{self.path_down_aie_to_mem}" begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if internal_right == 1:
            self.mem_animations_svg += f'<circle id="mem_circle_east{self.index}" class="mem_animations" cx="{self.internal_mem_x1}" cy="{self.internal_mem_y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M0, 0, \
                        l{self.internal_mem_x2}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'
        if internal_left == 1:
            self.mem_animations_svg += f'<circle id="mem_circle_west{self.index}" class="mem_animations" cx="{self.internal_mem_x1+ self.internal_mem_x2}" cy="{self.internal_mem_y1}" r="3" fill="none">\
                    <animateMotion dur="{duration}s" repeatCount="indefinite"\
                        path="M0, 0, \
                        l{-self.internal_mem_x2}, 0\
                        " begin="{delay}s"/>\
                    <animate attributeName="fill"\
                        values="none;{color}" dur="0.0001s" begin="{delay}s" fill="freeze"\
                        />\
                    </circle>\n'

    def _generate_tile_svg(self):
        self.aie_tile_svg = self.get_aie_svg() # reverse order to draw tiles from the bottom and the
        self.memory_connection_svg = self.get_memory_connections_svg()
        self.ic_connection_svg = self.get_ic_connections_svg()
        self.ic_animation_svg = self.get_ic_animations_svg()
        self.memory_animation_svg = self.get_memory_animations_svg()
        self.aie_buffers_svg = self.get_aie_buffers_svg()

        self.aie_svg = self.aie_tile_svg \
            +self.memory_connection_svg \
            +self.memory_animation_svg \
            +self.ic_connection_svg \
            +self.ic_animation_svg \
            +self.aie_kernel_svg \
            +self.aie_buffers_svg \
            +self.hide_tile_svg

    def generate_image(self, filename="aie_tile.svg", show_border=0):
        self._generate_tile_svg()

        self.generate_svg_file(self.aie_svg, filename, show_border)


class MemTile(Tile, Image):
    MAX_BUFFERS = 16
    def __init__(self, row, col):
        super().__init__(row, col, "mem_tile", config.mem_tile_bg_color, config.mem_container_bg_color)
        self.mem_svg = ""
        self.number_of_buffers = 0
        self.buffers_svg = ""

        self.mem_tile_memory = MemoryBox(
            "mem_tile_memory",
            self.index,
            self.x + config.mem_tile_memory_offset_x,
            self.y + config.mem_tile_memory_offset_y,
            config.mem_tile_memory_width,
            config.mem_tile_memory_height,
            "white",
            stroke_width=1,
            text1="Memory Tile",
            text2="Data movers",
            font_size=config.font_size,
            text_color="black",
            max_buffers = self.MAX_BUFFERS,
            obj_class="mem_tile_buffers"
        )
        self.mem_path_east_x1 = self.x + config.local_memory_offset_x + config.local_memory_width
        self.mem_path_east_y1 = self.y + config.aie_container_offset_y + int(config.aie_container_height / 2)
        self.mem_path_east_x2 = config.tile_width + config.tile_gap - config.aie_container_width+2*config.container_offset

        self.mem_path_west_x1 = self.x + config.aie_container_offset_x
        self.mem_path_west_y1 = self.mem_path_east_y1
        self.mem_path_west_x2 = -(self.mem_path_east_x2)

    def add_memory_outlines(self):
         self.mem_tile_memory.add_memory_outlines(self.MAX_BUFFERS)

    def draw_memory_connections(self, east : bool = False, west : bool = False, color="red"):
        line = ""

        if east:
            line += f'<line id="east_mem_tile_connection{self.index}" class="mem_tile_mem_connections" x1="{self.mem_path_east_x1}" y1="{self.mem_path_east_y1}" \
                x2="{self.mem_path_east_x1+self.mem_path_east_x2}" y2="{self.mem_path_east_y1}" \
                stroke="{color}" stroke-width="1" fill="none" />\n'
        if west:
            line += f'<line id="west_mem_tile_connection{self.index}" class="mem_tile_mem_connections" x1="{self.mem_path_west_x1}" y1="{self.mem_path_west_y1}" \
                x2="{self.mem_path_west_x1+self.mem_path_west_x2}" y2="{self.mem_path_west_y1}" \
                stroke="{color}" stroke-width="1" fill="none" />\n'

        self.memory_connections = line

    def add_ic_animation(self, diagonal_to_tile=0, diagonal_from_tile=0,
                         north=0, south=0, duration=2, delay=0, color="red"):
        self.mem_tile_memory.pulse_text = (diagonal_to_tile or diagonal_from_tile) or self.mem_tile_memory.pulse_text
        # Interface Tile in the memory tile does not move data to east/west
        super().add_ic_animation(diagonal_to_tile, diagonal_from_tile, north,
                                 south, east=0, west=0, duration=duration,
                                 delay=delay, color=color)

    def add_single_tile_ic_animation(self, north_up=0, north_down=0,
                                     duration=2, delay=0, color="red"):
        # Interface Tile in the memory tile does not move data to east/west
        super().add_single_tile_ic_animation(north_up, north_down,
                                             west_left=0, west_right=0,
                                             duration=duration, delay=delay,
                                             color=color)

    def get_buffers_svg(self):
        return self.buffers_svg

    def get_mem_tile_svg(self):
        self.tile_svg = self.get_tile_svg() \
            + self.mem_tile_memory.box \
            + self.hide_tile_svg
            #+ self.tile_ic_connection_svg \
            #+ self.tile_ic_animation_svg \
            #+ self.memory_connections \
            #+ self.ic_animations_svg \
            #+ self.buffers_svg \

        return self.tile_svg

    def _generate_tile_svg(self):
        self.tile_svg = self.get_mem_tile_svg()
        self.ic_connection_svg = self.get_ic_connections_svg()
        self.ic_animation_svg = self.get_ic_animations_svg()
        self.buffers_svg = self.get_buffers_svg()

        self.mem_tile_svg = self.tile_svg \
            +self.memory_connections \
            +self.ic_animation_svg \
            +self.ic_connection_svg \
            +self.buffers_svg \
            +self.hide_tile_svg

    def add_buffer(self, color, duration=2, start_empty=1, delay=0, color2=""):
        self.buffers_svg += \
            self.mem_tile_memory.add_buffer(color, duration, start_empty,
                                            delay, color2)

    def generate_image(self, filename="mem_tile.svg", show_border=0):
        self._generate_tile_svg()
        self.generate_svg_file(self.mem_tile_svg, filename, show_border)


class IfTile(Tile, Image):
    def __init__(self, row, col):
        super().__init__(row, col, "if_tile", config.if_tile_bg_color, config.if_container_bg_color)

        self.if_svg = ""
        self.external_interface_svg = ""

        self.if_box = Box(
            "if_dmabox",
            self.index,
            self.x + config.if_dmabox_offset_x,
            self.y + config.if_dmabox_offset_y,
            config.if_dmabox_width,
            config.if_dmabox_height,
            "white",
            stroke_width=1,
            text1="Interface Tile",
            text2="Data movers",
            font_size=config.font_size,
            text_color="black"
        )
        self.mem_path_south_x1 = self.x+config.aie_container_offset_x+config.aie_container_width/2
        self.mem_path_south_y1 = self.y+config.aie_container_offset_y+config.aie_container_height
        #self.mem_path_south_x2 = 0
        self.mem_path_south_y2 = config.ic_offset+config.tile_gap

    def get_if_tile_svg(self):
        self.ic_connection_svg = self.get_ic_connections_svg()
        self.ic_animation_svg = self.get_ic_animations_svg()
        self.memory_animation_svg = self.get_memory_animations_svg()
        self.if_svg = self.get_tile_svg() \
            + self.if_box.box \
            + self.ic_connection_svg \
            + self.external_interface_svg\
            + self.ic_animation_svg \
            + self.memory_animation_svg \
            + self.hide_tile_svg
        return self.if_svg

    def add_ic_animation(self, diagonal_to_tile=0, diagonal_from_tile=0, duration=2, delay=0, color="red"):
        self.if_box.pulse_text = True

        super().add_ic_animation(diagonal_to_tile=diagonal_to_tile, diagonal_from_tile=diagonal_from_tile, \
                                 north=0, south=0, east=0, west=0, duration=duration, delay=delay, color=color)


    def add_dma_animation(self, south_up=0, south_down=0, color="blue",
                          duration=2, delay=0):
        self.if_box.pulse_text = True
        return super().add_single_tile_mem_animation(north_up=0,
                                                     north_down=0,
                                                     south_up=south_up,
                                                     south_down=south_down,
                                                     east_left=0,
                                                     east_right=0,
                                                     west_right=0,
                                                     west_left=0,
                                                     color=color,
                                                     duration=duration,
                                                     delay=delay)

    def add_single_tile_mem_animation(self, **kargs):
        #Method not supported for this class
        pass

    def generate_image(self, filename="if_tile.svg", show_border=0):
        _ = self.get_if_tile_svg()

        self.generate_svg_file(self.if_svg, filename, show_border)

    def draw_external_memory_connections(self, south=0, color="black"):
        line = ""
        if south:
            path = f'M {self.mem_path_south_x1}, {self.mem_path_south_y1}, \
                l0, {self.mem_path_south_y2}'
            self.external_interface_svg = f'<path id="south_mem_connection{self.index}" stroke="{color}" stroke-width="1" fill="none"\
                d="{path}"\
                />\n'

class SystemMemory:
    def __init__(self, x : int = 0, y : int = 0, width : int = 50,
                 max_buffers: int = 4):
        self.x = x
        self.y = y
        self.width = width
        self._max_buffers = max_buffers
        self._buffers_svg = ""

        self._sysmem = MemoryBox(
            "main_memory",
            0,
            self.x,
            self.y + config.tile_height + 20,
            self.width,
            config.aie_box_height,
            "white",
            stroke_width=1,
            text1="System Memory",
            font_size=config.font_size+4,
            text_color="black",
            max_buffers=self._max_buffers,
            obj_class="sysmem_buffers"
        )
        self._sysmem.add_memory_outlines(self._max_buffers)

    def add_buffer(self, color, duration=2, start_empty=1, delay=0, color2=""):
        self._buffers_svg += \
            self._sysmem.add_buffer(color, duration, start_empty, delay,
                                    color2)

    @property
    def svg(self) -> str:
        """Return SVG code"""
        return self._sysmem.box + self._buffers_svg


class RyzenAiColumn(Image):

    def __init__(self, rows=4, cols=1, if_tile=1, mem_label: bool = False):
        super(Image, self).__init__()
        self.rows=rows
        self.cols=cols

        self.aie_tiles = []
        self.mem_tiles = []
        self.if_tiles = []

        self.if_tile = if_tile
        self.mem_label = mem_label

        self.aie_tiles_svg = ""
        self.mem_tiles_svg = ""
        self.if_tiles_svg = ""

        self.ic_connections_svg = ""
        self.ic_animations_svg = ""

        self.memory_connections_svg = ""
        self.memory_animations_svg = ""

        self.aie_kernels_svg = ""
        self.buffers_svg = ""
        self.svg_content = ""
        self.hide_tiles_svg = ""

        self.key_kernels = []
        self.key_width = 0
        self.key_svg = ""
        self.show_sysmem = False

        i = 0
        for j in range(self.rows):
            # Draw AIEs
            self.aie_tiles.append(AieTile(j, i, True))  # Instantiate AIE tiles

        for j in range(self.rows):
            index = j * self.cols + i
            # Draw IC connections
            if i == self.cols - 1:
                self.aie_tiles[index].draw_ic_connections(north=0, south=1, east=0, west=0)
            else:
                self.aie_tiles[index].draw_ic_connections(north=0, south=1, east=1, west=0)
            self.aie_tiles[index].add_memory_outlines()

        for j in range(self.rows - 1):
            index = j * self.cols + i
            # draw data memory connections for top and middle
            self.aie_tiles[index].draw_memory_connections(south_mem_to_aie=1,
                                                          south_aie_to_mem=1)

        j+=2
        self.mem_tiles.append(MemTile(j, i))
        self.mem_tiles[0].draw_ic_connections(north=0, south=1, east=0, west=0)

        if self.if_tile:
            j+=1
            self.if_tiles.append(IfTile(j, i))
            self.if_tiles[0].draw_ic_connections(north=0, south=0, east=0, west=0)
            self.if_tiles[0].draw_external_memory_connections(south=1, color="black")

        sysmem_width = self.mem_tiles[self.cols - 1].x - self.mem_tiles[0].x + config.tile_width

        self.sysmem = SystemMemory(self.aie_tiles[0].x, self.if_tiles[0].y,
                                   sysmem_width, max_buffers=4)

    def generate_column_svg(self):
        self.svg_content = ""
        self.hide_tiles_svg = ""
        for j in range(self.rows):
            for i in range(self.cols):
                index = j*self.cols+i
                if not self.mem_label:
                    self.aie_tiles[index].add_memory_outlines()
                self.aie_tiles_svg += self.aie_tiles[index].get_aie_svg()
                self.aie_kernels_svg += self.aie_tiles[index].get_aie_kernels_svg()
                self.buffers_svg += self.aie_tiles[index].get_aie_buffers_svg()
                self.memory_connections_svg += self.aie_tiles[index].get_memory_connections_svg()
                self.ic_connections_svg += self.aie_tiles[index].get_ic_connections_svg()
                self.ic_animations_svg += self.aie_tiles[index].get_ic_animations_svg()
                self.memory_animations_svg += self.aie_tiles[index].get_memory_animations_svg()
                self.hide_tiles_svg += self.aie_tiles[index].get_hide_tile_svg()

        for i in range(self.cols):
            self.mem_tiles[i].add_memory_outlines()
            self.mem_tiles_svg += self.mem_tiles[i].get_mem_tile_svg()
            self.buffers_svg += self.mem_tiles[i].get_buffers_svg()
            self.ic_connections_svg += self.mem_tiles[i].get_ic_connections_svg()
            self.ic_animations_svg += self.mem_tiles[i].get_ic_animations_svg()

        if self.if_tile:
            i = 0
            self.if_tiles_svg += self.if_tiles[i].get_if_tile_svg()

        sysmem_svg = self.sysmem.svg if self.show_sysmem else ""

        self.svg_content = self.aie_tiles_svg \
            +self.mem_tiles_svg\
            +self.if_tiles_svg\
            +self.ic_connections_svg\
            +self.ic_animations_svg\
            +self.memory_connections_svg\
            +self.memory_animations_svg\
            +self.aie_kernels_svg\
            +self.buffers_svg\
            +self.hide_tiles_svg\
            +self.key_svg\
            +sysmem_svg

        return self.svg_content

    def generate_image(self, filename="ryzenai_column.svg"):
        _ = self.generate_column_svg()
        self.generate_svg_file(self.svg_content, filename, show_sysmem=self.show_sysmem)

    def generate_key(self, kernels):
        self.show_key()
        key_item_height = 50
        text_offset = 108

        x = config.tile_width
        y =  (config.y_start
                + (config.tile_height + config.tile_gap) * self.rows
                + (config.tile_height + config.tile_gap)
                + config.tile_height/2
        )

        if self.key_width == 0:
            warnings.warn("Width for key is 0. Did you run .show_key()?")
        else:
            index = 1000 # Start key index at 1000 or this may conflict with other buffers
            key_kernels = []
            kernel_names = []
            for [kernel_name, color] in kernels:
                y -= key_item_height
                index += 1
                duration = 0
                delay = 0
                key_kernels.append(Kernel(x,y,index, color, duration, delay))
                self.key_svg += key_kernels[-1].get_kernel_svg()
                self.key_svg += f'<text class="text" x="{x+text_offset}" y="{y+105}" >{kernel_name}</text>'

    def draw_system_memory(self):
        """Draw system memory"""
        self.show_sysmem = True

    def hide_compute_tile(self):
        for ctile in self.aie_tiles:
            ctile.hide_tile()

    def show_compute_tile(self):
        for ctile in self.aie_tiles:
            ctile.show_tile()

    def hide_memory_tile(self):
        for mtile in self.mem_tiles:
            mtile.hide_tile()

    def show_memory_tile(self):
        for mtile in self.mem_tiles:
            mtile.show_tile()

    def hide_interface_tile(self):
        for itile in self.if_tiles:
            itile.hide_tile()

    def show_interface_tile(self):
        for itile in self.if_tiles:
            itile.show_tile()


class RyzenAiArray(Image):
    """
    A class representing a Ryzen AI Array. Inherits from the Image class.

    Args:
        rows (int): Number of rows in the array (default is 4).
        cols (int): Number of columns in the array (default is 5).
        ns_mem_connections (bool): Whether to include memory connections as north south, \
        or more complicated path from memory (North) to AIE (south) (default is False).
        remove_if_tile (bool): Whether to remove first interface tile (for RyzenAI) (default is True).

    """

    def __init__(self, rows=4, cols=5, ns_mem_connections=False, remove_if_tile=True):
        super().__init__()
        self.rows=rows
        self.cols=cols

        self.aie_tiles = []
        self.mem_tiles = []
        self.if_tiles = []

        self.aie_tiles_svg = ""
        self.mem_tiles_svg = ""
        self.if_tiles_svg = ""

        self.ic_connections_svg = ""
        self.ic_animations_svg = ""

        self.memory_connections_svg = ""
        self.memory_animations_svg = ""

        self.aie_kernels_svg = ""
        self.buffers_svg = ""
        self.svg_content = ""
        self.app_box = ""

        self.hide_all_tiles_svg = ""
        self.hide_aie_tiles_svg = ""
        self.hide_mem_tiles_svg = ""
        self.hide_if_tiles_svg = ""
        self.hide_mem_connections_svg = ""
        self.hide_interconnect_svg = ""

        self.hidden_tile_svg = "" # variable to collect hiding SVG for individual tiles
        self.hidden_tiles_svg = "" # variable for hiding classes of tiles in array

        self.show_sysmem = False

        self.key_width = 0

        if remove_if_tile:
            self.start_col=1 # Skip first tile
        else:
            self.start_col=0

        for j in range(self.rows):
            for i in range(self.cols):
                # Draw AIEs
                self.aie_tiles.append(AieTile(j, i, True))  # Instantiate AIE tiles


        for j in range(self.rows):
            for i in range(self.cols):
                index = j * self.cols + i
                self.aie_tiles[index].add_memory_outlines()
                # Draw IC connections and add animations to all
                if i == (self.cols - 1):
                    self.aie_tiles[index].draw_ic_connections(north=0, south=1, east=0, west=0)
                else:
                    self.aie_tiles[index].draw_ic_connections(north=0, south=1, east=1, west=0)

        for j in range(self.rows):
            for i in range(self.cols):
                index = j * self.cols + i
                east = self.cols > 1

                # Determine the location of the current tile
                is_top_right = index == (self.cols - 1)
                is_bottom_left = index == (self.rows - 1) * self.cols
                is_bottom_right = index == self.rows * self.cols - 1
                is_right = (index + 1) % self.cols == 0
                is_bottom_middle = index > ((self.rows - 1) * self.cols - 1)

                # Determine the memory connections based on tile location
                if is_top_right:
                    if ns_mem_connections:
                        self.aie_tiles[index].draw_memory_connections(south=1)
                    else:
                        self.aie_tiles[index].draw_memory_connections(south_mem_to_aie=1)
                elif is_bottom_left:
                    self.aie_tiles[index].draw_memory_connections(east=east)
                elif is_bottom_right:
                    self.aie_tiles[index].draw_memory_connections()  # Draw internal connection
                elif is_right:
                    if ns_mem_connections:
                        self.aie_tiles[index].draw_memory_connections(south=1)
                    else:
                        self.aie_tiles[index].draw_memory_connections(south_mem_to_aie=1)
                elif is_bottom_middle:
                    self.aie_tiles[index].draw_memory_connections(east=east)
                else:
                    if ns_mem_connections:
                        self.aie_tiles[index].draw_memory_connections(south=1, east=east)
                    else:
                        self.aie_tiles[index].draw_memory_connections(south_mem_to_aie=1, east=east)

        j+=1
        for i in range(self.cols):
            mem_tile = MemTile(j, i)
            self.mem_tiles.append(mem_tile)

            north_connection = 0
            south_connection = 1 if (self.start_col == 0 or i > 0) else 0
            east_connection = 1 if i < self.cols - 1 else 0
            west_connection = 0

            mem_tile.draw_ic_connections(
                north=north_connection,
                south=south_connection,
                east=east_connection,
                west=west_connection
        )

        j+=1
        for i in range(self.start_col,self.cols):
            self.if_tiles.append(IfTile(j, i))
            self.if_tiles[i-self.start_col].draw_ic_connections(north=0, south=0, east=0, west=0)

        sysmem_width = self.mem_tiles[self.cols - 1].x - self.mem_tiles[0].x + config.tile_width

        self.sysmem = SystemMemory(self.aie_tiles[0].x, self.if_tiles[0].y,
                                   sysmem_width, max_buffers=4*self.cols)

    def generate_svg(self):
        for j in range(self.rows):
            for i in range(self.cols):
                index = j*self.cols+i
                self.aie_tiles_svg += self.aie_tiles[index].get_aie_svg() # reverse order to draw tiles from the bottom and the left
                self.aie_kernels_svg += self.aie_tiles[index].get_aie_kernels_svg()
                self.buffers_svg += self.aie_tiles[index].get_aie_buffers_svg()
                self.memory_connections_svg += self.aie_tiles[index].get_memory_connections_svg()
                self.hidden_tile_svg += self.aie_tiles[index].get_hide_tile_svg()
                self.ic_connections_svg += self.aie_tiles[index].get_ic_connections_svg()
                self.ic_animations_svg += self.aie_tiles[index].get_ic_animations_svg()
                self.memory_animations_svg += self.aie_tiles[index].get_memory_animations_svg()

        self.mem_tile_memory_connections_svg = ""
        for i in range(self.cols):
            self.mem_tiles[i].add_memory_outlines()
            self.mem_tiles_svg += self.mem_tiles[i].get_mem_tile_svg()
            self.buffers_svg += self.mem_tiles[i].get_buffers_svg()
            self.ic_connections_svg += self.mem_tiles[i].get_ic_connections_svg() # This will cause a duplicate of the ic_connections; need to fix
            self.ic_animations_svg += self.mem_tiles[i].get_ic_animations_svg()
            self.memory_connections_svg += self.mem_tiles[i].get_memory_connections_svg()
            self.memory_animations_svg += self.mem_tiles[i].get_memory_animations_svg()

        for i in range(self.cols-self.start_col):
            self.if_tiles_svg += self.if_tiles[i].get_if_tile_svg()

        self.update_hidden_tiles() # Update self.hidden_tiles_svg

        sysmem_svg = self.sysmem.svg if self.show_sysmem else ""

        self.svg_content = self.hidden_tiles_svg\
            +self.aie_tiles_svg \
            +self.mem_tiles_svg\
            +self.if_tiles_svg\
            +self.memory_connections_svg\
            +self.mem_tile_memory_connections_svg\
            +self.hidden_tile_svg \
            +self.ic_connections_svg\
            +self.ic_animations_svg\
            +self.memory_animations_svg\
            +self.aie_kernels_svg\
            +self.buffers_svg\
            +self.app_box\
            +sysmem_svg

        return self.svg_content

    def generate_svg_image(self, filename="ryzenai_array.svg", debug=0):

        _ = self.generate_svg()
        self.generate_svg_file(self.svg_content, filename, show_sysmem=self.show_sysmem)

    def draw_app_box(self, col : int = 1, width : int = 1, text : str = ""):
        """ Add a box around the column to represent an application
        
        Parameters
        ----------
        col: index of the column
        width: number of columns
        text: application name
        """
        center_y1 = 4
        x = config.tile_gap*0.75 + (config.tile_width + config.tile_gap) * col
        y = 5
        box_width = (config.tile_width + 10) * width + 10 * (width-1)
        font_size = config.font_size
        text_width = config.get_text_width(text, font_size)
        center_x = x + (box_width - text_width)/2

        box = f'<g> <rect id="text" class="appbox" ' \
              f'width="{box_width}" '\
              f'height="{(config.tile_height + config.tile_gap)*6.05}" x="{x}" y="{y}" ' \
              f'stroke="black" stroke-width="4" '\
              f'fill="white" opacity="0.5"/> </g>\n' \
              f'<g> <rect id="text" class="appbox" ' \
              f'width="{box_width}" '\
              f'height="{font_size*1.2}" x="{x}" y="{y}" ' \
              f'stroke="black" stroke-width="4" '\
              f'fill="black"/> </g>\n' \
              f'<g>'\
              f'<text id="text_app" '\
              f'class="apptext" x="{x}" y="{y+10}" '\
              f'fill="white" '\
              f'font-family="Arial, Helvetica, sans-serif"> '\
              f'<tspan x="{center_x}" y="{center_y1}" dy="1em" '\
              f'font-size="{font_size}">{text} </tspan>'\
              f'</text></g>\n'

        self.app_box += box

    def draw_system_memory(self):
        """Draw system memory"""
        self.show_sysmem = True

    def hide_tiles(self, opacity=0.25):
        self.hide_all_tiles_svg = f'\
        <style>\
            .border, .aie_tile_container, .mem_tile_container, .if_tile_container, .aie_box, .local_memory_box, .mem_tile_memory, .if_dmabox, .aie_tile_buffers, .mem_tile_buffers, .aie_tile, .if_tile, .mem_tile, .aie_tile_text, mem_tile_text, .if_tile_text, .kernel {{ \
                opacity: {opacity}; \
            }}\
        </style>'

    def hide_aie_tiles(self, opacity=0.25):
        '''Hide AIE tiles only. '''
        self.hide_aie_tiles_svg = f'\
        <style>\
            .border, .aie_tile_container, .aie_box, .local_memory_box, .aie_tile_buffers, .aie_tile, .aie_tile_text, .kernel {{ \
                opacity: {opacity}; \
            }}\
        </style>'

    def show_aie_tiles(self):
        self.hide_aie_tiles_svg = ""

    def hide_mem_tiles(self, opacity=0.25):
        self.hide_mem_tiles_svg = f'\
        <style>\
            .border, .mem_tile_container, .mem_tile_memory, .mem_tile, .mem_tile_text, .mem_tile_buffers {{ \
                opacity: {opacity}; \
            }}\
        </style>'

    def show_mem_tiles(self):
        self.hide_mem_tiles_svg = ""

    def hide_if_tiles(self, opacity=0.25):
        self.hide_if_tiles_svg = f'\
        <style>\
            .border, .if_tile, .if_dmabox, .if_tile_container, .if_tile_text {{ \
                opacity: {opacity}; \
            }}\
        </style>'

    def show_if_tiles(self):
        self.hide_if_tiles_svg = ""

    def show_tiles(self):
        self.hide_all_tiles_svg = ""
        self.hide_compute_tiles_svg = ""
        self.hide_mem_tiles_svg = ""
        self.hide_if_tiles_svg = ""

    def hide_memory_connections(self, opacity=0.25):
        self.hide_mem_connections_svg = f'\
        <style>\
            .mem_connections {{ \
                opacity: {opacity}; \
            }}\
        </style>'

    def hide_interconnect(self, opacity=0.25):
        self.hide_interconnect_svg = f'\
        <style>\
            .interconnect, .ic_animations, .interconnect_box {{ \
                opacity: {opacity}; \
            }}\
        </style>'

    def show_interconnect(self):
        self.hide_interconnect_svg = ""

    def update_hidden_tiles(self):
        self.hidden_tiles_svg = self.hide_all_tiles_svg + self.hide_aie_tiles_svg + self.hide_mem_tiles_svg + self.hide_if_tiles_svg + self.hide_mem_connections_svg + self.hide_interconnect_svg


class Box:
    """Define a box with"""
    def __init__(self, id_name: str, index:int=0, x:int=0, y:int=0,
                 width:int=0, height:int=0, color:str="#000000",
                 stroke_color:str="black", stroke_width:int=1,
                 text1:str="", text2:str="", font_size:int=1,
                 text_color:str="", pulse_text:bool=0, show_label:bool=True):
        self.id_name = id_name
        self.index= index
        self.x = x
        self.y = y
        self.center_x = x + width / 2
        self.center_y = y + height / 2
        self.width = width
        self.height = height
        self.color = color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.text1 = text1
        self.text2 = text2
        self.font_size = font_size
        self.text_color = text_color
        self.pulse_text = pulse_text
        self.show_label = show_label
        self._append = ""
        self._buffers = ""

    @property
    def box(self):
        """Return SVG text for a box"""
        animate = ""
        border_width = self.stroke_width
        if self.pulse_text:
            animate = f'<animate attributeName="stroke-width" dur="0.5s" \
                      repeatCount="indefinite" values="{self.stroke_width-1}; \
                      {self.stroke_width+1};{self.stroke_width-1}" />\n'
            border_width = ""

        if self.show_label:
            text1 = self.text1
            text2 = self.text2
        else:
            text1 = text2 = ""

        center_x1 = self.center_x - config.get_text_width(text1, self.font_size)/2
        center_x2 = self.center_x - config.get_text_width(text2, self.font_size)/2
        center_y1 = self.center_y - \
            self.font_size * 0.7 -  ((self.font_size / 2) if text2 else 0)
        center_y2 = center_y1 + self.font_size

        # Stroke width needs to be empty for the animation on stroke to work
        box = f'<g> <rect id="{self.id_name+str(self.index)}" '\
              f'class="{self.id_name}" width="{self.width}" '\
              f'height="{self.height}" x="{self.x}" y="{self.y}" ' \
              f'stroke="{self.stroke_color}" stroke-width="{border_width}" '\
              f'fill="{self.color}"/>{animate} </g>\n'
        label = f'<g> <text id="text_{self.id_name+str(self.index)}" '\
                f'class="{self.id_name}_text" x="{self.x}" y="{self.y}" '\
                f'fill="{self.text_color}" '\
                f'font-family="Arial, Helvetica, sans-serif"> '\
                f'<tspan x="{center_x1}" '\
                f'y="{center_y1}" dy="1em" '\
                f'font-size="{self.font_size}">{text1} </tspan>'\

        if text2:
            label += f'<tspan x="{center_x2}" '\
                     f'y="{center_y2}" dy="1em" '\
                     f'font-size="{self.font_size}">{text2} </tspan>'

        label += '</text></g>\n'
        return self._buffers + box +  label + self._append

    @property
    def append(self):
        pass

    @append.setter
    def append(self, text):
        self._append += text


class MemoryBox(Box):
    def __init__(self, id_name: str, index: int = 0, x: int = 0,
                    y: int = 0, width: int = 0, height: int = 0,
                    color: str = "#000000", stroke_color: str = "black",
                    stroke_width: int = 1, text1: str = "", text2: str = "",
                    font_size: int = 1, text_color: str = "",
                    pulse_text: bool = 0, show_label: bool = True,
                    max_buffers : int = 8, obj_class : str = "aie_tile_buffers"):
        super().__init__(id_name, index, x, y, width, height, None,
                            stroke_color, stroke_width, text1, text2,
                            font_size, text_color, pulse_text, show_label)
        self._max_buffers = max_buffers
        self.buffer_count = 0
        self.obj_class = obj_class

    def add_memory_outlines(self, buffers : int = 8):
        """Add buffer outline"""
        for i in range(self._max_buffers):
            buffer_offset = (self.width  / self._max_buffers) * i
            self._buffers += f'<g> '\
                f'<rect id="memory_outline_{i}_{self._max_buffers}" '\
                f'x="{self.x + buffer_offset}" '\
                f'y="{self.y}" '\
                f'height="{self.height}" '\
                f'width="{self.width/self._max_buffers}" '\
                f'style="stroke:#f0f0f0; fill:#fefefe; stroke-opacity:0.5"> '\
                '</rect> </g>\n'

    def add_buffer(self, color, duration=2, start_empty=1, delay=0, color2=""):
        """ Add animated buffer in a memory
        Parameters
        ----------
        color: str
            Color of the buffer

        Optional
        --------
        duration: int
            Duration of the animation in seconds
        start_empty: bool
            Buffer starts animation empty
        delay: int
            Animation starts after `delay` seconds
        color2: str
             Color used to empty the buffer
        """
        self.show_label = False

        if self.buffer_count >= self._max_buffers:
            warnings.warn("Error: MAX number of buffers reached")
            return

        color3 = color
        if color2 != "":
            color3 = color2

        cfill = color
        cempty = color3
        if not start_empty:
            cfill = color3
            cempty = color

        if duration == 0:
            duration2 = 0.0001
            repeat = "frezee"
        else:
            duration2 = duration*2
            repeat = "indefinite"

        if start_empty==0:
            values = f'"{self.height};0;{self.height}"'
        else:
            values = f'"0;{self.height};0"'
        buffer_offset = (self.width / self._max_buffers) * self.buffer_count
        buffer_width = self.width/self._max_buffers
        rotate_x = self.x +(self.width / self._max_buffers)/2+ buffer_offset
        rotate_y = self.y + self.height
        buffer = f'<g transform="rotate(180, {rotate_x}, {rotate_y}) "> '\
                    f'<rect id="buffer{self.index}_{self.buffer_count}" '\
                    f'class="{self.obj_class}" x="{self.x + buffer_offset}" '\
                    f'y="{rotate_y}" height="{self.height}" '\
                    f'width="{buffer_width}" style="stroke:none; fill:none"> '\
                    f'<animate attributeName="height" values={values} '\
                    f'keyTimes="0;0.5;1" dur="{duration2}s" begin="{delay}s" '\
                    f'fill="freeze" repeatCount="{repeat}" /> '\
                    f'<animate attributeName="fill" '\
                    f'values="none;{cfill};{cfill};{cempty};{cempty}" '\
                    f'keyTimes="0; 0.00001; 0.5; 0.50001; 1" '\
                    f'dur="{duration2}s" begin="{delay}s" fill="freeze" '\
                    f'repeatCount="{repeat}" '\
                    f'/></rect></g>\n'
        self.buffer_count += 1
        return buffer
