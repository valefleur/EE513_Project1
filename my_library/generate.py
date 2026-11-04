import os
import sys
import math
import rawpy
# from PIL import Image
# import pillow_heif
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Transfer functions from Prof. James McNames' psu_imaging.py (EE 513 Introduction to Image Processing).
import psu_imaging as psu

from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

def capture_edge(sigma=1.0, angle_deg=5.0, h=256, w=128, black_level=psu.BLACK_LEVEL, white_level=psu.WHITE_LEVEL, exposure=0.55, seed=0, noisy=True, supersample_scale=1):
    """
    A monochrome slanted-edge capture, returned in linear light.
    Note: Code borrowed from Prof. James McNames' 04_measure_your_camera.ipynb and modified as needed for homework 2.
    """
    scene = psu.slanted_edge(h*supersample_scale, w*supersample_scale, angle_deg=angle_deg)
    blurred = cv.GaussianBlur(scene, (0, 0), sigmaX=sigma*supersample_scale, sigmaY=sigma*supersample_scale, borderType=cv.BORDER_REPLICATE)
    if not noisy:
        return blurred
    raw = psu.expose_and_digitize(blurred, np.random.default_rng(seed), exposure=exposure)
    return (raw.astype(float) - black_level) / (white_level - black_level)


def cos4_map(h, w, bin_factor=2):
    """cos^4 falloff for an h x w image binned `bin_factor` from the 4032 x 3024 raw."""
    f_px = cam_spec.focal_mm * 1000 / (cam_spec.pitch_um * bin_factor)
    y, x = np.mgrid[0:h, 0:w]
    r2 = (x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2
    return (1 + r2 / f_px**2) ** -2