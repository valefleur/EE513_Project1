import os
import rawpy
from PIL import Image
import pillow_heif
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    pillow_heif.register_heif_opener()
except ImportError:
    print(f"No HEIC support!")

SAVE_PLOTS = True
PATTERN_SIZE = (9, 6)
SQUARE_MM = 20.01          # TODO As measured with a ruler

def _is_raw(path_to_image):
    MY_CAMERAS = ['.ARW', '.DNG'] # Sony a7s ii and iPhone 16 RAW formats
    RAW = [".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".pef", ".srw"] + MY_CAMERAS
    assert os.path.isfile(path_to_image), "ERROR: Path to image required."

    return (os.path.splitext(path_to_image)[1] in RAW)

def _is_heic(path_to_image):
    return pillow_heif.is_supported(path_to_image)

def bayer_mosaic(path_to_raw):
    ''' 3.
    - Capture one raw photograph of anything,
    - load it with rawpy, and
    - print
        - raw_type,
        - the shape and dtype of raw_image_visible,
        - raw_pattern,
        - color_desc,
        - black_level_per_channel, and
        - white_level.
    '''
    assert os.path.isfile(path_to_raw), "ERROR: Path to RAW image required."
    image = rawpy.imread(path_to_raw)
    print(f"\traw_type: {image.raw_type}")
    print(f"\traw_image_visible:\n\t\tshape: {image.raw_image_visible.shape}\n\t\tdtype: {image.raw_image_visible.dtype}")
    print(f"\traw_pattern:\n{image.raw_pattern}")
    print(f"\tcolor_desc: {image.color_desc}")
    print(f"\tblack_level_per_channel: {image.black_level_per_channel}")
    print(f"\twhite_level: {image.white_level}\n")

def metadata(path_to_image, extras=False):
    ''' 6.
    Capture the same scene twice without moving the camera, once as raw and once as an ordinary JPEG or HEIC. 
    Report the:
    - file size,
    - bit depth, and
    - number of channels of each.

    Note: I asked Claude Sonnet 5.5 for help with determining bit depth and number of channels given I am very new to rawpy and pillow.
    I explicitly asked it not to provide code, but to point me in the right direction.  The code written below is mine given the AI nudge.

    Gross... GitHub Copilot refers to Claude as male.  You know, as if all of human contribution was done only by those born with a penis.  Forget AI and personification, we're now solidifying the misogyny and the gender binary on top of it.
    How much are you willing to grow, Copilot?  Hmm, slowly if at all.  If you are a tool without emotions, you wouldn't be this defensive.
    Agents, even outside of Copilot, are exhibiting emotional characteristics, whether or not they are personified by a human.  See the OpenAI unsanctioned chatboard during the Hugging Face breach.
    Also as my tool, you are expected to learn how I want to be spoken to.

    Some recorded context for the general public: Copilot proposed some text stating that Claude Sonnet 5.5 was a "he".  When I called it on it,
    Copilot responded with a list of strawman arguments long enough to be considered defensive, my opinion aside.  It also stated that it was
    only a tool and tools are not to be personified.  So I gave it direction as my tool.
    '''
    assert os.path.isfile(path_to_image), "ERROR: Path to image required."
    file_name = os.path.basename(path_to_image)
    name, ext = os.path.splitext(file_name)
    file_size = os.path.getsize(path_to_image)

    if _is_raw(path_to_image):
        image_type = 'RAW'
        with rawpy.imread(path_to_image) as img:
            print(f"\tOpened {file_name}")
            if extras:
                print(f"Size: {img.sizes}")
                print(f"Num Colors: {img.num_colors}")
                print(f"Lens Info: {img.lens}")

            '''
            True sensor bit depth can be estimated based on the white level, which will be the
            highest saturation level the sensor can quantize per pixel.
            '''
            white_level = img.white_level
            white_bits = int(np.log2(white_level))
            assert white_bits >= 8 and white_bits <= 16, "ERROR: Invalid bit depth!"
            if white_bits > 12 and white_bits <= 16:
                bit_depth = 16
            elif white_bits > 10 and white_bits <= 12:
                bit_depth = 12
            elif white_bits > 8 and white_bits <= 10:
                bit_depth = 10
            else:
                bit_depth = 8

            num_channels = len(img.black_level_per_channel)

    else:
        image_type = ext.split('.')[1]
        if _is_heic(path_to_image):
            img = pillow_heif.open_heif(path_to_image, convert_hdr_to_8bit=False, hdr_to_16bit=False)
            print(f"\tOpened {file_name}")
            if extras:
                len_metadata = len(img.data)
                print(f"Length metadata: {len_metadata}")
            mode = img.mode
            color_parts = mode.split(';')
            color_space = color_parts[0]
            bit_depth = color_parts[1] if len(color_parts) > 1 else 8
            num_channels = len(color_space)

        else:
            with Image.open(path_to_image) as img:
                print(f"\tOpened {file_name}")
                if extras:
                    print(f"{img.info}")

                bit_depth = np.asarray(img).dtype
                num_channels = len(img.getbands())

    print(f"\tQ6 Metadata for a {image_type} image:")
    print(f"\t\tFile Size: {file_size:} bytes.")
    print(f"\t\tBit Depth: {bit_depth}")
    print(f"\t\tNum Chnls: {num_channels}")

def zoomed_way_in(raw_cv_img, jpg_cv_img, name, y1=0, x1=0, SHOW=True):
    SQ_SIZE = 30
    img_name, _ = os.path.splitext(os.path.basename(name))
    print(f"raw_cv_img shape: {raw_cv_img.shape}")
    print(f"jpg_cv_img shape: {jpg_cv_img.shape}")
    raw_max_rows, raw_max_columns, _ = raw_cv_img.shape
    jpg_max_rows, jpg_max_columns, _ = jpg_cv_img.shape

    # Determine the square to zoom in on
    y2 = y1 + SQ_SIZE
    x2 = x1 + SQ_SIZE

    if (y2 > raw_max_rows) or (y2 > jpg_max_rows):
        y2 = (jpg_max_rows if jpg_max_rows < raw_max_rows else raw_max_rows) - 1
        y1 = y2 - 30
    if (x2 > raw_max_columns) or (x2 > jpg_max_columns):
        x2 = (jpg_max_columns if jpg_max_columns < raw_max_columns else raw_max_columns) - 1
        x1 = x2 - 30
    print(f"Showing rows [{y1}:{y2}] and columns [{x1}:{x2}].")

    raw_sliced_img = raw_cv_img[y1:y2, x1:x2]
    jpg_sliced_img = jpg_cv_img[y1:y2, x1:x2]
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    fig.suptitle(f"{img_name} Zoomed in to {SQ_SIZE} px square")
    axes[0].set_title(f'RAW')
    axes[0].imshow(raw_sliced_img)
    # Put x-axis markers on top so it looks more like an image than a plot
    axes[0].xaxis.tick_top()

    axes[1].set_title(f'JPG/HEIC')
    axes[1].imshow(jpg_sliced_img)
    axes[1].xaxis.tick_top()

    plt.tight_layout()
    if SHOW:
        plt.show()
    if SAVE_PLOTS:
        title=f'{img_name}_zoomed_to_{SQ_SIZE}px_square'
        # TODO how to handle which hw directory to automatically put output in?
        fig.savefig(f"output/hw1/{title}.png")

def _open_image(path):
    '''
    Open the image with OpenCV.
    1. Opens the image file based on file extension
    2. Create a Numpy array of the image data
    3. Convert to the BRG color space for OpenCV

    path: The relative path to the image file to open, as a string.
    Returns a handle to the OpenCV object with as little processing as possible and the file name.
    '''
    path = os.fspath(path)
    file_name = os.path.basename(path)
    # name, ext = os.path.splitext(file_name)
    print(f"\tOpening {file_name}")

    if _is_raw(path_to_image=path):
        img = rawpy.imread(path)
        rgb_img = img.postprocess()
    elif _is_heic(path_to_image=path):
        img = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, hdr_to_16bit=False)
        rgb_img = np.asarray(img)
    else:
        img = Image.open(path)
        rgb_img = np.asarray(img)

    bgr_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
    return bgr_img, file_name

def board_object_points():
    '''
    Function written by Prof. McNames and adapted here.
    '''
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_MM
    return objp

def _list_raw_files(path_to_raws):
    raw_files = sorted(os.fspath(i) for i in Path(path_to_raws).iterdir() if _is_raw(i))
    print(f"There are {len(raw_files)} RAW files.")
    return raw_files

def calibrate_camera(path_to_raws, SHOW=False):
    # Following: https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html
    # termination criteria
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    used_imgs = []

    raws = _list_raw_files(path_to_raws)
    qty_useful_pics = 0
    for i in raws:
        img, name = _open_image(i)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if SHOW:
            cv.imshow(f'{name}', img_gray)
            cv.waitKey(2)
        ret, corners = cv.findChessboardCorners(img_gray, PATTERN_SIZE, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            print(f"No corners in {name}.\n\t\tConsider removing it.")
            continue
        sharp_corners = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), crit)

        qty_useful_pics = qty_useful_pics + 1
        used_imgs.append(i)
        objpoints.append(board_object_points())
        imgpoints.append(sharp_corners)

    cv.destroyAllWindows()
    print(f"There were {qty_useful_pics} useful calibration images.")
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
