# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from PIL import ImageFont, Image, ImageDraw
import platform

ROWS = 4
COLS = 5

# offset from the top and side

x_start = 20
y_start = 20

# AIE, Mem and Interface (if) tile
tile_width = 180
tile_height = 140

interconnect_box_width = 40
interconnect_box_height = interconnect_box_width

# offset ic from tile border
ic_offset = 10

aie_container_width = 120
aie_container_height = 60

mem_tile_height = tile_height + aie_container_height

# offset inner tiles from container border
container_offset = 7

# gap between tiles
tile_gap = 20
text_x_offset = 10
text_y_offset = 5

# Derived positional values

## Local

aie_container_height = aie_container_width / 2
aie_container_offset_x = tile_width - aie_container_width - ic_offset
aie_container_offset_y = tile_height - aie_container_height - ic_offset

diagonal_offset = aie_container_offset_y - interconnect_box_width - ic_offset

aie_box_width = aie_container_width / 2 - container_offset * 2
aie_box_height = aie_box_width

aie_box_offset_x = (
    aie_container_offset_x + (aie_container_width - aie_box_width * 2) / 4
)
aie_box_offset_y = aie_container_offset_y + (aie_container_height - aie_box_height) / 2

local_memory_offset_x = (
    aie_box_offset_x + aie_box_width + (aie_container_width - aie_box_width * 2) / 2
)
local_memory_offset_y = aie_box_offset_y

local_memory_width = aie_box_width
local_memory_height = aie_box_height

mem_tile_memory_width = aie_container_width - container_offset*2
mem_tile_memory_height = aie_box_height # aie_box_height*2

mem_container_height = aie_container_height + aie_box_height

mem_tile_memory_offset_x = aie_container_offset_x + container_offset
mem_tile_memory_offset_y = aie_container_offset_y + container_offset
ic_width = tile_width + tile_gap - interconnect_box_width
ic_height = tile_height + tile_gap - interconnect_box_height

if_dmabox_width = mem_tile_memory_width
if_dmabox_height = aie_box_height # mem_tile_memory_height

if_dmabox_offset_x = mem_tile_memory_offset_x
if_dmabox_offset_y = mem_tile_memory_offset_y

# Effort made to make it easy for color blind people

background = "white"
highlight = "white"
dim = "#9FAFA1"
notes = "#3C4A3E"

blue = "#56B4E9"
orange = "#E69F00"
dark_blue = "#0072B2" #"#0072B2"
red = "#C00000"
yellow = "#F0E442"
green = "#009E73"
artichoke = "#8F9779"
pink = "#CC79A7" # "#CC79A7"
dark_orange = "#D55E00"
dark_pink = "#DC267F"
purple = "#4B0092"
orchid = "#AF69EE"
lilac = "#E0C2FF"
magenta = "#FF00FF"
violet = "#8F00FF"

light_orange = "#F7E2B2"
light_red = "#C36D6D"
light_blue = "#B2D4E8"
light_pink = "#E6BBD1"


aie_tile_bg_color = "#C2E9FF" # "#EAEAEA"
mem_tile_bg_color = "#E0FFC2" #"#EAEAEA" #"#99CC00"
if_tile_bg_color = lilac #"#EAEAEA" #"#CCCCFF"

aie_container_bg_color = "#ffffff" #"#C2E9FF" #"#CCECFF"
mem_container_bg_color = "#ffffff" # "#E0FFC2" #"#99CC00"
if_container_bg_color = "#ffffff" # lilac #"#CCCCFF"

interconnect_color = "#DBBCBD"

font_size = 13

def get_tile_x(col: int) -> int:
    """Get X coordinate for a AIE tile"""
    return x_start + (tile_width + tile_gap) * col


def get_tile_y(row: int) -> int:
    """Get Y coordinate for a AIE tile"""
    return y_start + (tile_height + tile_gap) * row


def get_text_width(text : str = None, font_size : int = font_size):
    """Get with of text for a particular font_size"""
    if platform.system() == "Linux":
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    else:
        font_path = "C:\\Windows\\Fonts\\arial.ttf"

    font = ImageFont.truetype(font_path, int(font_size))
    image = Image.new('RGB', (1, 1), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    text_width = draw.textlength(text, font=font)
    return text_width
