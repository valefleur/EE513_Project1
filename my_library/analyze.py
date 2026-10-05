import rawpy
import os
from PIL import Image
import pillow_heif
import numpy as np

try:
    pillow_heif.register_heif_opener()
except ImportError:
    print(f"No HEIC support!")

def _is_raw(path_to_image):
    RAW = ['.ARW', '.DNG'] # Sony a7s ii and iPhone 16 RAW formats
    assert os.path.isfile(path_to_image), "ERROR: Path to image required."

    return (os.path.splitext(path_to_image)[1] in RAW)

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

    bits = np.arange(8, 16, 2)
    if _is_raw(path_to_image):
        image_type = 'RAW'
        with rawpy.imread(path_to_image) as img:
            print(f"Opened {file_name}")
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
        if pillow_heif.is_supported(path_to_image):
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

                bit_depth = img.mode
                num_channels = len(img.getbands())

    print(f"\tQ6 Metadata for a {image_type} image:")
    print(f"\t\tFile Size: {file_size} bytes.")
    print(f"\t\tBit Depth: {bit_depth}")
    print(f"\t\tNum Chnls: {num_channels}")