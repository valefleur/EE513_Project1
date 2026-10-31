import os
import sys
import math
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
CAMERAS = {'sony': 'Sony_a7sii', 'iphone': 'iPhone_16ProMax'}
PATTERN_SIZE = (9, 6)
SQUARE_MM = 20.01          # As measured with a ruler

def _is_raw(path_to_image):
    MY_CAMERAS = ['.arw', '.dng'] # Sony a7s ii and iPhone 16 RAW formats
    RAW = [".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".pef", ".srw"] + MY_CAMERAS
    assert os.path.isfile(path_to_image), "ERROR: Path to image required."

    return (os.path.splitext(path_to_image)[1].lower() in RAW)

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

def zoomed_way_in(raw_cv_img, jpg_cv_img, name, y1=0, x1=0, camera=None, SHOW=False):
    assert (camera is not None) and (camera in CAMERAS.keys()), "ERROR: Must have camera type to store output!"
    print(f"camera is [{camera}]")
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
        fig.savefig(f"output/hw1/07_zoomed/{CAMERAS.get(camera)}/{title}.png")

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
    # print(f"\tOpening {file_name}")

    if _is_raw(path_to_image=path):
        img = rawpy.imread(path)
        rgb_img = img.postprocess(user_flip=0, no_auto_bright=True, use_camera_wb=True)
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
    raw_file_paths = sorted(os.fspath(i) for i in Path(path_to_raws).iterdir() if _is_raw(i))
    print(f"There are {len(raw_file_paths)} RAW files.")
    return raw_file_paths


def contact_sheet(image_paths, camera=None, output=None, question=None, labels=None, SHOW=False):
    assert (camera is not None) and (camera in CAMERAS.keys()), "ERROR: Must have camera type to store output!"
    assert (labels is None) or (len(labels) == len(image_paths)), f"ERROR: If labels [{len(labels)}] are provided, they must match the number of images [{len(image_paths)}]."
    HEIGHT = 100
    WIDTH = 100
    thumbs = []
    print(f'image_paths[0]: {image_paths[0]}')
    for i in image_paths:
        # print(f'\tOpening {i}')
        img, name = _open_image(i)
        small_img = cv.cvtColor(cv.resize(img, (WIDTH, HEIGHT)), cv.COLOR_BGR2RGB)
        thumbs.append(small_img)
        del img

    # if less than 20 images, use 4 columns
    # If there are more than 20 images, use 5 columns
    # If there are more than 40 images, use 10 columns
    if len(thumbs) <= 20:
        qty_cols = 4
    elif len(thumbs) <= 30:
        qty_cols = 5
    else:
        qty_cols = 10
    # qty_rows = int(len(thumbs) / qty_cols) + (1 if len(thumbs) % qty_cols else 0) # less efficient than math.ceil()
    qty_rows = math.ceil(len(thumbs) / qty_cols)
    # print(f"There is room for {qty_rows}*{qty_cols} = {qty_rows*qty_cols} thumbnails [and we have {len(thumbs)}].")

    fig, axes = plt.subplots(qty_rows, qty_cols, figsize=(10 * qty_cols, 8 * qty_rows))
    axes = np.atleast_1d(axes).ravel()
    # print(f'There are {len(axes)} of type [{axes.dtype}]')
    fig.suptitle(f"Contact Sheet for {CAMERAS.get(camera)}")
    for f, thumb, label in zip(axes, thumbs, labels if labels is not None else [None] * len(thumbs)):
        f.imshow(thumb)
        f.xaxis.tick_top()
        if label is not None:
            f.set_title(label, fontsize=8)
    for f in range(len(thumbs), qty_rows*qty_cols):
        axes[f].axis('off')

    # close the opened images to free memory
    for img in thumbs:
        del img

    # reduce the whitespace between the subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    if SHOW:
        plt.show()
    if SAVE_PLOTS:
        title=f'{CAMERAS.get(camera)}_contact_sheet'
        if output:
            path = f"{output}/{question+'_' if question is not None else ''}{title}.png"
        else:
            # TODO update HW1 code and output file structure to use more generic version above
            path = f"output/hw1/10_contact_sheet/{CAMERAS.get(camera)}/{title}.png"
        fig.savefig(path)


def plot_residuals(real_points, image_points, rms, K, distortion, rvecs, tvecs, per_view, shape, camera=None, SHOW=True):
    # Note: Code borrowed from Prof. James McNames with slight tweaks
    assert (camera is not None) and (camera in CAMERAS.keys()), "ERROR: Must have camera type to store output!"
    res, pos = [], []
    for op, ip, rv, tv in zip(real_points, image_points, rvecs, tvecs):
        proj, _ = cv.projectPoints(op, rv, tv, K, distortion)
        res.append(ip.reshape(-1, 2) - proj.reshape(-1, 2)) # Claude suggested this
        # res.append(ip.reshape(-1, 2) - ip.reshape(-1, 2)) # McNames does this
        pos.append(ip.reshape(-1,2))
        # collect ip.reshape(-1, 2) and res, then plt.quiver(x, y, res_x, res_y) across all images
    res, pos = np.vstack(res), np.vstack(pos)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.quiver(pos[:, 0], pos[:, 1], res[:, 0], res[:, 1],
            np.linalg.norm(res, axis=1), cmap="viridis",
            angles="xy", scale_units="xy", scale=0.002, width=0.003)
    a1.add_patch(plt.Rectangle((0, 0), *shape, fill=False, ec="#213921", lw=1.2))
    a1.set_xlim(-200, shape[0] + 200); a1.set_ylim(shape[1] + 200, -200)
    a1.set_aspect("equal"); a1.grid(alpha=0)
    a1.set_title("residuals by position, drawn 500x", fontsize=11)
    a1.xaxis.tick_top()

    a2.plot(np.arange(1, len(per_view) + 1), per_view.ravel(), "o-",
            color="#6d8d24", lw=1.5, ms=5)
    a2.axhline(rms, color="#213921", ls="--", lw=1.3, label=f"overall {rms:.3f} px")
    a2.set_xlabel("view"); a2.set_ylabel("per-view RMS (px)")
    a2.set_title("which capture is dragging the fit", fontsize=11); a2.legend(fontsize=9)
    a2.xaxis.tick_top()

    plt.tight_layout()
    if SHOW:
        plt.show()
    if SAVE_PLOTS:
        title=f'Residuals'
        fig.savefig(f"output/hw1/{CAMERAS.get(camera)}_{title}.png")


def edge_deviation(pts, pattern=PATTERN_SIZE, side='first_col'):
    """
    Fit a line through one outer column of corners.
    Return the points, the line, and signed perpendicular deviations (px).
    """
    cols, rows = pattern # PATTERN_SIZE = (9, 6): 9 corners across, 6 down
    grid = np.asarray(pts, np.float32).reshape(rows, cols, 2)   # corners come back row-major
    edge = np.ascontiguousarray(grid[:, 0, :] if side == 'first_col' else grid[:, -1, :])

    vx, vy, x0, y0 = cv.fitLine(edge, cv.DIST_L2, 0, 0.01, 0.01).ravel()   # unit direction + a point on the line
    dev = (edge[:, 0] - x0) * vy - (edge[:, 1] - y0) * vx                  # signed perpendicular distance
    return edge, (vx, vy, x0, y0), dev


def undistort(img, K, distortion, corners=None, camera=None, SHOW=True, DEBUG=True):
    assert (camera is not None) and (camera in CAMERAS.keys()), "ERROR: Must have camera type to store output!"

    img_to_undistort, img_name = _open_image(img)
    rbg_img_to_undistort = cv.cvtColor(img_to_undistort, cv.COLOR_BGR2RGB)

    print(f"Undistorting: {img_name} which has size: {rbg_img_to_undistort.shape[:2]}")
    newK = None
    if DEBUG:
        h, w = rbg_img_to_undistort.shape[:2]
        newK, roi = cv.getOptimalNewCameraMatrix(K, distortion, (w, h), alpha=1)
    undistorted = cv.undistort(rbg_img_to_undistort, K, distortion, None, newK)

    pts_raw = pts_und = None
    if corners is not None:
        pts_raw = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        P = newK if newK is not None else K
        pts_und = cv.undistortPoints(pts_raw, K, distortion, P=P).reshape(-1, 2)
        pts_raw = pts_raw.reshape(-1, 2)
    else:
        print(f'Corners is empty in undistort()')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = "Undistorting"
    fig.suptitle(f"{title}")
    axes[0].imshow(rbg_img_to_undistort)
    axes[0].set_title(f'What the Camera Saw')
    axes[0].xaxis.tick_top()
    x = y = []
    last_on_left = -1 - PATTERN_SIZE[0]
    if pts_raw.any():
        # x.append(pts_raw[0, 0])
        # y.append(pts_raw[0, 1])
        # x.append(pts_raw[last_on_left, 0])
        # y.append(pts_raw[last_on_left, 1])
        # axes[0].plot(x, y, '-', color='yellow') # for every row, plot column 0
        # axes[0].scatter(x, y, s=12, c='red', edgecolors='white', linewidths=0.4)
        # axes[0].plot(pts_raw[:, 0], pts_raw[:, 1], '-', color='yellow') # for every row, plot column 0
        # axes[0].scatter(pts_raw[:, 0], pts_raw[:, 1], s=12, c='red', edgecolors='white', linewidths=0.4)
        edge, (vx, vy, x0, y0), raw_dev = edge_deviation(pts_raw)
        t = (edge - [x0, y0]) @ [vx, vy]                      # position of each point along the line
        p0 = np.array([x0, y0]) + t.min() * np.array([vx, vy])
        p1 = np.array([x0, y0]) + t.max() * np.array([vx, vy])
        axes[0].plot([p0[0], p1[0]], [p0[1], p1[1]], 'c-', lw=1.5)
        axes[0].scatter(edge[:, 0], edge[:, 1], s=40, facecolors='none', edgecolors='cyan')
        axes[0].set_xlabel(f"Max deviation: {np.abs(raw_dev).max():.2f} px\nRMS: {np.sqrt(np.mean(raw_dev**2)):.2f} px")

    axes[1].imshow(undistorted)
    axes[1].set_title(f'The Undistorted View{' Full Frame' if DEBUG else ''}')
    axes[1].xaxis.tick_top()
    if pts_und.any():
        # axes[1].plot(pts_und[:, 0], pts_und[:, 1], '-', color='blue') # for every row, plot column 0
        # axes[1].scatter(pts_und[:, 0], pts_und[:, 1], s=12, c='purple', edgecolors='white', linewidths=0.4)
        edge, (vx, vy, x0, y0), und_dev = edge_deviation(pts_und)
        t = (edge - [x0, y0]) @ [vx, vy]                      # position of each point along the line
        p0 = np.array([x0, y0]) + t.min() * np.array([vx, vy])
        p1 = np.array([x0, y0]) + t.max() * np.array([vx, vy])
        axes[1].plot([p0[0], p1[0]], [p0[1], p1[1]], 'c-', lw=1.5)
        axes[1].scatter(edge[:, 0], edge[:, 1], s=40, facecolors='none', edgecolors='cyan')
        axes[1].set_xlabel(f"Max deviation: {np.abs(und_dev).max():.2f} px\nRMS: {np.sqrt(np.mean(und_dev**2)):.2f} px")

    plt.tight_layout()
    if SHOW:
        plt.show()
    if SAVE_PLOTS:
        fig.savefig(f"output/hw2/03_{camera}_{title}.png")

    # Problem 3
    problem_num = "03"
    print(f"\n**** Problem {problem_num} ****")
    print(f"Raw Deviation: Max: {np.abs(raw_dev).max():.2f} px, RMS: {np.sqrt(np.mean(raw_dev**2)):.2f} px")
    print(f"Undistorted Deviation: Max: {np.abs(und_dev).max():.2f} px, RMS: {np.sqrt(np.mean(und_dev**2)):.2f} px")

    return rbg_img_to_undistort, undistorted, pts_raw, pts_und


def calibrate_camera(path_to_raws, camera=None, oneoff=None, SHOW=False):
    assert (camera is not None) and (camera in CAMERAS.keys()), "ERROR: Must have camera type to store output!"
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
    sizes = [] # TODO GitHub issue #3, confirm camera didn't flip to portrait mode when taking calibration images

    oneoff_corners = []
    if oneoff:
        oneoff_name = os.path.basename(oneoff)
        print(f'***** Oneoff image name: {oneoff_name}')

    raws = _list_raw_files(path_to_raws)
    contact_sheet(raws, camera)
    qty_useful_pics = 0
    for i in raws:
        img, name = _open_image(i)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # print(f"Image {i} is {name}.")
        sizes.append(img_gray.shape[::-1])
        if SHOW:
            cv.imshow(f'{name}', img_gray)
            cv.waitKey(2)
        if 1:
            ret, corners = cv.findChessboardCorners(img_gray, PATTERN_SIZE, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
            if not ret:
                print(f"No corners in {name}.  Removing it.")
                continue
            sharp_corners = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), crit)
        else:
            ret, sharp_corners = cv.findChessboardCornersSB(img_gray, PATTERN_SIZE, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
            if not ret:
                print(f"No corners in {name}.  Removing it.")
                continue
        if oneoff and (name in [oneoff_name]):
            oneoff_corners.append(sharp_corners)
            print(f'---> Oneoff image: {oneoff_name}')

        qty_useful_pics = qty_useful_pics + 1
        used_imgs.append(i)
        objpoints.append(board_object_points())
        imgpoints.append(sharp_corners)

        cv.drawChessboardCorners(img, PATTERN_SIZE, sharp_corners, ret)
        if SHOW:
            cv.imshow('Corners', img)
            cv.waitKey(2)
        if SAVE_PLOTS:
            cv.imwrite(f'output/hw1/11_corners/{CAMERAS.get(camera)}/{name}_corners_shown.png', img)

    cv.destroyAllWindows()
    # assert len(sizes) == 1, f"Mixed image sizes: {sizes}"
    print(f"There were {qty_useful_pics} useful calibration images.")
    # return objpoints, imgpoints, img_gray.shape[::-1], used_imgs
    if not qty_useful_pics:
        print(f"Not enough quality pictures to calibrate the camera!")
        sys.exit()
    flgs = None
    # flgs = cv.CALIB_FIX_K3
    # flgs = cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
    rms, K, dist, rvecs, tvecs, sd_int, sd_ext, per_view = cv.calibrateCameraExtended(objpoints, imgpoints, img_gray.shape[::-1], None, None, flags=flgs)

    plot_residuals(objpoints, imgpoints, rms, K, dist, rvecs, tvecs, per_view, sizes[0], camera=camera)

    print(f'RMS: {rms}')
    print(f'K:\n{K}')
    print(f'Image Size: {sizes[0]}')
    print(f'Image Center: {(sizes[0][0]/2,sizes[0][1]/2)}')
    print(f'dist:\n{dist}')
    print(f'rvecs length: {len(rvecs)}')
    print(f'tvecs length: {len(tvecs)}')
    print(f'sd_int shape: {sd_int.shape}')
    print(f'sd_ext shape: {sd_ext.shape}')
    print(f'per_view shape: {per_view.shape}')

    # Improved diagnostics
    # Show the reprojection RMS for files whose RMS is large
    RMS_LIMIT = 1.5
    for p, err in zip(used_imgs, per_view.ravel()):
        if err > RMS_LIMIT:
            print(f"Consider tossing {os.path.basename(p)}: RMS is {err:.3f} px")
    print("intrinsic std devs (fx, fy, cx, cy, k1, k2, p1, p2, k3):\n", sd_int.ravel())

    return K, dist, (oneoff_corners[0] if len(oneoff_corners) > 0 else None)


def generate_side_by_side(image_0, image_1, title=None, SHOW=False):
    image_0, image_0_name = _open_image(image_0)
    image_1, image_1_name = _open_image(image_1)

    rgb_img_0 = cv.cvtColor(image_0, cv.COLOR_BGR2RGB)
    rgb_img_1 = cv.cvtColor(image_1, cv.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if title:
        fig.suptitle(f"{title}")
    axes[0].imshow(rgb_img_0)
    axes[0].xaxis.tick_top()
    axes[0].set_title(f'{image_0_name}')
    axes[1].imshow(rgb_img_1)
    axes[1].xaxis.tick_top()
    axes[1].set_title(f'{image_1_name}')

    plt.tight_layout()
    if SHOW:
        plt.show()
    if SAVE_PLOTS:
        title_alt = f"{image_0_name}+{image_1_name}"
        fig.savefig(f"output/hw1/06_RAW_vs_Processed/{title if title else title_alt}.png")