import cv2
import os
from collections import namedtuple

import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from object_detection.utils import dataset_util


def save_image(stroke_list,
               image_name,
               directory,
               width=100,
               my_dpi=50):
    """Draw with a list of strokes and save the image as png"""
    fig = plt.figure(figsize=(width / my_dpi, width / my_dpi),
                     dpi=my_dpi,
                     frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    min_x = 255
    max_x = 0
    min_y = 255
    max_y = 0
    for stroke in stroke_list:
        x = stroke[0]
        y = stroke[1]
        min_x = min(min_x, min(x))
        max_x = max(max_x, max(x))
        min_y = min(min_y, min(y))
        max_y = max(max_y, max(y))
        plt.plot(x, y, c='k')

    plt.xlim((min_x, max_x))
    plt.ylim((min_y, max_y))
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(directory, image_name),
                bbox_inches='tight',
                pad_inches=0,
                dpi=my_dpi)
    plt.close(fig)
