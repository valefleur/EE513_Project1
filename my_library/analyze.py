import rawpy

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
    image = rawpy.imread(path_to_raw)
    print(f"\traw_type: {image.raw_type}")
    print(f"\traw_image_visible:\n\t\tshape: {image.raw_image_visible.shape}\n\t\tdtype: {image.raw_image_visible.dtype}")
    print(f"\traw_pattern:\n{image.raw_pattern}")
    print(f"\tcolor_desc: {image.color_desc}")
    print(f"\tblack_level_per_channel: {image.black_level_per_channel}")
    print(f"\twhite_level: {image.white_level}\n")

