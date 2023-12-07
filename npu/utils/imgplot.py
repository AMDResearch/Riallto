# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_plot(img_src: np.ndarray, img_dst: np.ndarray = None,
               alpha: bool = False) -> None:
    """Plots the `img_src` using matplotlib

    Parameters:
    -----------
    img_src: np.ndarray
        Primary image to plot.
    img_dst: np.ndarray [Optional]
        Secondary image to plot. If provided it will be labeled as Processed
        Image
    alpha: bool [Optional]
        Use alpha channel for visualization.
    """

    multiplot = (img_dst is not None)
    sizew = 20 if multiplot else 9
    sizeh = 10 if multiplot else 6
    _, axes = plt.subplots(1, (2 if multiplot else 1), figsize=(sizew, sizeh))
    if not multiplot:
        axes = [axes]

    for idx, (image, ax) in enumerate(zip([img_src, img_dst], axes)):
        nimg = image
        if image.ndim == 3 and image.shape[2] == 4 and not alpha:
            nimg = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        ax.imshow(np.array(nimg), cmap=(None if nimg.ndim == 3 else 'gray'))
        ax.axis("off")
        if multiplot:
            ax.set_title('Original Image' if idx == 0 else 'Processed Image')
