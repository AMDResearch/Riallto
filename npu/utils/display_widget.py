# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from IPython.display import display
import ipywidgets as widgets
import cv2
import numpy as np


class DisplayImage:
    """ Simple class for displaying images/video in Jupyter notebooks.

    Attributes
    ----------
    _image_widget: widgets.Image
        reference to the widgets.Image object.
    _button_widget : widgets.Button
        reference to the button widget used to stop the video feed.
    exit : bool
        set by the button widget and read by the display widget to stop video.
    """

    def __init__(self, resize: int = 1):
        """ returns a DisplayImage object """
        self._image_widget = widgets.Image(format='png')
        self._display()
        self._button_widget = widgets.Button(description='Stop')
        self._button_widget.on_click(self._stop_video)
        display(self._button_widget)
        self.exit = False
        self._resize = resize

    def _display(self) -> None:
        """ Creates the display widget """
        display(self._image_widget)

    def _stop_video(self, event=None) -> None:
        """On button press this function is called to set the exit attribute and stop the video"""

        self.exit = True
        self._button_widget.unobserve_all()
        self._button_widget.disabled = True

    def frame(self, value) -> None:
        """ Sets the current image on the widget """
        rows, cols, _ = value.shape
        frame = cv2.resize(value, (cols//self._resize, rows//self._resize))
        _, img = cv2.imencode('.png', frame)

        img_array = np.array(img)
        self._image_widget.value = memoryview(img_array.tobytes())
