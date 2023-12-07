# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import threading
import cv2
import numpy as np
from npu.runtime import AppRunner
from npu.utils.display_widget import DisplayImage
from typing import Dict

class ImageLooper720p():
    """ 
    Wrapper class that allows to feed in the same 720p image into
    an NPU application in a loop. 

    This enables the experimentation of changing different RTPs and 
    seeing how they effect the output image.

    Attributes
    ----------
    _disp : DisplayImage
        Reference to the ipython widget used to display the image
    _rtps : dict
        A dictionary describing what widgets to use for each rtp in the application and how to render them
    _xclbin : str
        string for the filepath to the xclbin location
    _img_file: str
        string for the filepath to the image that is going to be passed into the npu in a loop
    """

    def __init__(self, img:str, xclbin:str, rtps:Dict):
        self._disp = None
        self._rtps = rtps
        self._xclbin = xclbin
        self._img_file = img

    def _process_image(self)->None:
        """ Event loop that passes the image into the NPU and displays it. Checks for exit condition. """
        ubuff_in = self._app.allocate(shape=(720,1280,4), dtype=np.uint8)
        ubuff_out = self._app.allocate(shape=(720,1280,4), dtype=np.uint8)

        while True:
            ubuff_in[:] = self._img
            ubuff_in.sync_to_npu()
            self._app.call(ubuff_in, ubuff_out)
            ubuff_out.sync_from_npu()
            output_buffer = np.array(ubuff_out.bo.map(), dtype=np.uint8)
            out_img = output_buffer.reshape(720, 1280, 4)

            image = cv2.cvtColor(out_img, cv2.COLOR_BGRA2BGR)
            self._disp.frame(image)

            if self._disp.exit:
                del ubuff_in
                del ubuff_out
                self._app.__del__()
                del self._app
                break

    def start(self)->None:
        """ Creates an AppRunner object and starts a thread running the _process_image loop. """
        self._app = AppRunner(self._xclbin)
        self._disp = DisplayImage()
        self._img = cv2.imread(self._img_file)
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGBA)
        self._app.rtpwidgets(self._rtps)
        self._thread = threading.Thread(target=self._process_image)
        self._thread.start()
