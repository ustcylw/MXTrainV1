#! /usr/bin/env python
# coding: utf-8
import os, sys
from matplotlib import cm
from .singleton import SingletonType
import numpy as np


class Color(metaclass=SingletonType):

    def __init__(self, num_colors=50):
        self.num_colors = num_colors
        self.colors = self.get_n_colors(self.num_colors)
        self.color_idx = 0

    def get_n_colors(self, n, colormap="gist_ncar"):
        # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
        # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
        # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
        # For more options see:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # and https://matplotlib.org/users/colormaps.html

        colors = cm.get_cmap(colormap)(np.linspace(0, 1, n))
        # Randomly shuffle the colors
        np.random.shuffle(colors)
        # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
        # Also multiply by 255 since cm returns values in the range [0, 1]
        colors = colors[:, (2, 1, 0)] * 255
        return colors

    def get_color_iter(self):
        color = self.colors[self.color_idx % self.num_colors]
        self.color_idx += 1
        yield color
    
    def get_color(self, idx):
        return self.colors[idx % self.num_colors]
