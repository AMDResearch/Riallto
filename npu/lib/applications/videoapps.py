# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from enum import Enum
import os
import threading
import cv2
import inspect
import numpy as np
import colorsys
import time
import platform
import matplotlib.pyplot as plt
import ipywidgets as widget
from IPython.display import display
from typing import Union
from npu.utils.display_widget import DisplayImage
from npu.runtime import AppRunner


class pxtype(Enum):
    """Supported Image types"""
    RGBA = 0
    GRAY = 1


_resolutions = [(1080, 1920), (720, 1280)]
_int_slider_r = {'type': 'slider', 'min': 0, 'max': 255, 'name': 'Red'}
_int_slider_g = {'type': 'slider', 'min': 0, 'max': 255, 'name': 'Green'}
_int_slider_b = {'type': 'slider', 'min': 0, 'max': 255, 'name': 'Blue'}
_dropdown_thr = {
    "type": "dropdownpair",
    "options": [("BINARY", [0, 0]), ("BINARY_INV", [1, 0]),
                ("TRUNC", [2, 0]), ("TOZERO", [3, 0]),
                ("TOZERO_INV", [4, 0])],
    "name": "Type",
    "pair": "thresholdValue4"
    }


def _get_full_path(xclbin: str = None) -> str:
    binaries = os.path.dirname(os.path.abspath(__file__)) + '/binaries/'
    return os.path.abspath(os.path.join(binaries, xclbin))


def _find_closest_resolution(cam_h: int = None,
                             cam_w: int = None):
    """Find the closes available resolution

    Find the nearest resolution in each dimension and use it if it matches
    a valid resolution. Otherwise, pick the resolution that matches with the
    nearest width.
    """
    height = [x[0] for x in _resolutions]
    width = [x[1] for x in _resolutions]

    list_w = list(np.abs(np.array(width) - cam_w))
    idxw = list_w.index(np.min(list_w))
    list_h = list(np.abs(np.array(height) - cam_h))
    idxh = list_h.index(np.min(list_h))

    resolution = (_resolutions[idxh][1], _resolutions[idxw][0])

    if resolution not in _resolutions:
        resolution = _resolutions[idxw]

    return resolution


def _get_webcam_resolution(cap):
    """Get webcam resolution"""
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cam_h, cam_w


def _set_webcam_resolution(cap, height: int, width: int) -> bool:
    """Set webcam resolution

    Returns `True` if the camera supports the resolutions. Otherwise
    returns `False`.
    """
    _ = cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    _ = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)):
        return False
    return True


def _set_supported_webcam_resolution(cap) -> bool:
    """Try to set one of the supported webcam resolution

    Returns `True` if the camera supports one of the resolutions. Otherwise
    returns `False`.
    """
    for resolution in _resolutions:
        if _set_webcam_resolution(cap, resolution[0], resolution[1]):
            return True
    return False


class VideoApplication:
    """Wrapper class that allows to feed and visualize video stream from the NPU

    You must pass an xclbin and optionally the pixel type for the input and
    output images.

    Parameters
    ----------
    filename : str
        Path to the xclbin file
    videosource : [int, str]
        videosource webcam index or path to video file
    pxtype_in : pxtype
        Pixel type for the input image, either pxtype.RGBA or pxtype.GRAY
    pxtype_out : pxtype
        Pixel type for the output image, either pxtype.RGBA or pxtype.GRAY

    Returns
    -------
    VideoApplication
        Object that abstracts away the video handling from a webcam or
        video file to visualization
    """

    cam_w = None
    cam_h = None
    rtps = {}
    _camres = None
    _resize = None
    _cap = None

    def __init__(self, filename, videosource: Union[int, str] = 0,
                 pxtype_in: pxtype = pxtype.RGBA,
                 pxtype_out: pxtype = pxtype.RGBA):

        if not isinstance(pxtype_in, pxtype):
            raise ValueError(f"pxtype_in ({pxtype_in}) must be of the type "
                             f"{pxtype}")

        if not isinstance(pxtype_out, pxtype):
            raise ValueError(f"pxtype_out ({pxtype_out}) must be of the type "
                             f"{pxtype}")

        self._pxtype_in = pxtype_in
        self._pxtype_out = pxtype_out
        if not isinstance(videosource, (int, str)):
            raise ValueError("'videosource' must be either an index to a "
                             "webcam or a path to a video file")

        if isinstance(videosource, str) and not os.path.exists(videosource):
            raise FileNotFoundError(f"{videosource=} file does not exist")

        self._vsource = videosource
        if not self.cam_w or not self.cam_h:
            self._get_resolution(self._vsource)
        self.app = AppRunner(filename)
        self.thread = None
        self._disp = None

    def _get_resolution(self, videosource):
        if isinstance(videosource, int):
            if platform.system() == 'Linux':
                prop = cv2.CAP_V4L2
            elif videosource == 0:
                prop = cv2.CAP_MSMF
            else:
                prop = cv2.CAP_DSHOW
            self._cap = cv2.VideoCapture(videosource, prop)
            if not self._cap.isOpened():
                self._cap.release()
                raise IOError("Cannot read from webcam. Check if other "
                              "notebook is using the webcam.")
        else:
            self._cap = cv2.VideoCapture(videosource)

        self._resize = not _set_supported_webcam_resolution(self._cap)
        self.cam_h, self.cam_w = _get_webcam_resolution(self._cap)

        self._camres = self.cam_w, self.cam_h
        if self._resize:
            self.cam_h, self.cam_w = \
                _find_closest_resolution(self.cam_h, self.cam_w)

    def start(self):
        """Start the video processing"""
        time.sleep(0.3)
        ret, _ = self._cap.read()
        if not self._cap.isOpened() or not ret:
            self._cap.release()
            self._cleanup()
            raise IOError("Cannot read from webcam. Check if other notebook "
                          "is using the webcam.")

        self._disp = DisplayImage()
        self.app.rtpwidgets(self.rtps)
        self.thread = threading.Thread(target=self._process_video)
        self.thread.start()

    def _process_video(self):
        if self._pxtype_in == pxtype.RGBA:
            shape_in = (self.cam_h, self.cam_w, 4)
            typecolor = cv2.COLOR_BGR2RGBA
        else:
            shape_in = (self.cam_h, self.cam_w)
            typecolor = cv2.COLOR_BGR2GRAY

        if self._pxtype_out == pxtype.RGBA:
            shape_out = (self.cam_h, self.cam_w, 4)
        else:
            shape_out = (self.cam_h, self.cam_w)

        bo_in = self.app.allocate(shape=shape_in, dtype=np.uint8)
        bo_out = self.app.allocate(shape=shape_out, dtype=np.uint8)

        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret and isinstance(self._vsource, str):
                # restart video
                self._cap = cv2.VideoCapture(self._vsource)
                _, frame = self._cap.read()

            if self._resize:
                frame = cv2.resize(frame, (self.cam_w, self.cam_h))
            bo_in[:] = cv2.cvtColor(frame, typecolor)

            bo_in.sync_to_npu()
            self.app.call(bo_in, bo_out)
            bo_out.sync_from_npu()

            if self._pxtype_out == pxtype.RGBA:
                tmp = np.copy(bo_out[:, :, :3])
            else:
                tmp = np.copy(bo_out)

            self._disp.frame(tmp)

            if self._disp.exit:
                self._cap.release()

        self._cleanup()

    def stop(self):
        """Stop the video processing"""
        self._cap.release()

    @property
    def resolution(self):
        """Webcam video feed resolution"""
        return self._camres

    @property
    def videores(self):
        """Video feed resolution to/from the NPU"""
        return (self.cam_w, self.cam_h)

    def _cleanup(self):
        self.app.__del__()
        del self.app


class _PipecleanerVideoProcessing(VideoApplication):
    """Pipecleaner Video processing
    """
    def __init__(self, videosource: Union[int, str] = 0):
        self._get_resolution(videosource)
        filename = _get_full_path(f'color_threshold_v1_{self.cam_h}p.xclbin')
        super().__init__(filename, videosource)

    def _process_video(self):
        bo_in = np.zeros(shape=(self.cam_h, self.cam_w, 4), dtype=np.uint8)
        bo_out = np.zeros(shape=(self.cam_h, self.cam_w, 4), dtype=np.uint8)

        while self._cap.isOpened():
            _, frame = self._cap.read()
            bo_in[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            tmp = np.copy(bo_out[:, :, :3])
            self._disp.frame(tmp)

            if self._disp.exit:
                self._cap.release()

        self._cleanup()


class ColorThresholdVideoProcessing(VideoApplication):
    """Color Threshold Video processing
    """
    rtps = {'thresholdValue1': _int_slider_r,
            'thresholdValue2': _int_slider_g,
            'thresholdValue3': _int_slider_b,
            'thresholdType': _dropdown_thr}

    def __init__(self, videosource: Union[int, str] = 0):

        self._get_resolution(videosource)
        filename = _get_full_path(f'color_threshold_v1_{self.cam_h}p.xclbin')
        super().__init__(filename, videosource)


class ScaledColorThresholdVideoProcessing(VideoApplication):
    """Color Threshold Video processing
    """

    rtps = {'thresholdValue1': _int_slider_r,
            'thresholdValue2': _int_slider_g,
            'thresholdValue3': _int_slider_b,
            'thresholdType': _dropdown_thr}

    def __init__(self, videosource: Union[int, str] = 0):

        self._get_resolution(videosource)
        filename = _get_full_path(f'color_threshold_v2_{self.cam_h}p.xclbin')
        super().__init__(filename, videosource)


class _Filter2dOperator():
    EDGE = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int16)
    SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int16)
    SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.int16)
    SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.int16)
    SCHARR_X = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.int16)
    SCHARR_Y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.int16)
    PREWITT_X = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.int16)
    PREWITT_Y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.int16)


def _is_numpy_array(member):
    return isinstance(member, np.ndarray)


class EdgeDetectVideoProcessing(VideoApplication):
    """Edge Detect Video processing
    """
    rtps = {
        'alpha': {'type': 'slider', 'min': 0, 'max': 16384, 'name': 'edges'},
        'beta': {'type': 'slider', 'min': 0, 'max': 16384,
                 'name': 'color overlaid'},
        'thresholdValue': {'type': 'slider', 'min': 0, 'max': 255,
                           'name': 'threshold'}
    }

    def __init__(self, videosource: Union[int, str] = 0):

        self._get_resolution(videosource)
        filename = _get_full_path(f'edge_detect_{self.cam_h}p.xclbin')
        super().__init__(filename, videosource)

    def start(self):
        super().start()
        self._filter2d_widgets()

    def _filter2d_widgets(self):
        options = [x[0] for x in
                   inspect.getmembers(_Filter2dOperator(),
                                      predicate=_is_numpy_array)]
        self._dropdown = widget.Dropdown(
            options=options,
            description='Filter2D Operator',
            value=options[0],
        )
        self._dropdown.style.description_width = 'auto'
        self._dropdown.observe(lambda change:
                               self._filter2d_update(change.new), 'value')
        display(self._dropdown)

    def _filter2d_update(self, value):
        f2doperator = getattr(_Filter2dOperator, value)

        for r in range(3):
            for c in range(3):
                rtp = self.app.rtps['filter2dline_0'].get(f'weight_{r}_{c}')
                rtp['seq_ref'].value = f2doperator[r][c] * (2**12)


class ColorDetectVideoProcessing(VideoApplication):
    """Color Detect Video processing
    """

    rtps = {
        "thresholdValue1l": {
            "type": "hueslider",
            "min": 0,
            "max": 255,
            "rangehigh": "thresholdValue1u",
            'name': 'Hue range 0'
        },
        "thresholdValue2l": {
            "type": "hueslider",
            "min": 0,
            "max": 255,
            "rangehigh": "thresholdValue2u",
            'name': 'Hue range 1'
        }
    }

    def __init__(self, videosource: Union[int, str] = 0):

        self._get_resolution(videosource)
        filename = _get_full_path(f'color_detect_{self.cam_h}p.xclbin')
        super().__init__(filename, videosource)

    def start(self):
        super().start()

        # Create the HUE scale
        _, ax = plt.subplots(figsize=(5.9, 0.4))
        ax.set_aspect('equal', 'box')
        ax.set_yticks([])
        colors = [colorsys.hsv_to_rgb(hue / 255, 1, 1)
                  for hue in np.arange(0, 256, 1)]
        color_matrix = np.array([colors])
        ax.imshow(color_matrix, aspect='auto', extent=[0, 255, 0, 1])
        # Remove the  border
        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.annotate('Hue Scale', xy=(1.05, 0.5),
                    xycoords='axes fraction', va='center')
        ax.set_xticks([0, 32, 64, 96, 128, 160, 192, 224, 255])
        plt.show()


class DenoiseTPVideoProcessing(VideoApplication):
    """Denoising Task Parallel Video processing
    """
    def __init__(self, videosource: Union[int, str] = 0):

        self._get_resolution(videosource)
        filename = \
            _get_full_path(f'denoise_task_parallel_{self.cam_h}p.xclbin')
        super().__init__(filename, videosource, pxtype_out=pxtype.GRAY)


class DenoiseDPVideoProcessing(VideoApplication):
    """Denoising Data Parallel Video processing
    """
    def __init__(self, videosource: Union[int, str] = 0):

        self._get_resolution(videosource)
        filename = \
            _get_full_path(f'denoise_data_parallel_{self.cam_h}p.xclbin')
        super().__init__(filename, videosource, pxtype_out=pxtype.GRAY)
