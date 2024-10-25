# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from IPython.display import display
import ipywidgets as widgets
from io import BytesIO
import PIL


class DisplayImage:
    """ Simple class for displaying images/video in Jupyter notebooks.

    Attributes
    ----------
    _image_widget: widgets.Image
        reference to the widgets.Image object.
    _button_widget : widgets.Button
        reference to the button widget used to stop the video feed.
    exit : bool
        set by the button widget to stop the video feed
    """

    def __init__(self):
        """ returns a DisplayImage object """
        self._image_widget = widgets.Image(format='jpeg')
        self._display()
        self._button_widget = widgets.Button(description='Stop')
        self._button_widget.on_click(self._stop_video)
        display(self._button_widget)
        self.exit = False

    def _display(self) -> None:
        """ Creates the display widget """
        display(self._image_widget)

    def _stop_video(self, event=None) -> None:
        """ On button press this function stops the video feed"""
        self.exit = True
        self._button_widget.unobserve_all()
        self._button_widget.disabled = True

    def frame(self, value, scale: int = 1) -> None:
        """ Sets the current image on the widget """
        pil_image = PIL.Image.fromarray(value)
        if scale > 1:
            rows, cols, _ = value.shape
            pil_image = pil_image.resize((cols//scale, rows//scale))
        b = BytesIO()
        pil_image.save(b, format='jpeg')
        self._image_widget.value = b.getvalue()
