from PIL import Image
import os
import numpy as np
import pandas as pd

def clean_image_data(filepath, ids: pd.DataFrame):
    id_list = ids['id'].to_list()
    try:
        os.mkdir("cleaned_images")
        dirs = os.listdir(filepath)
        final_size = 512
        for item in dirs:
            if item[:-4] in id_list:
                im = Image.open(os.path.join(filepath, item))
                new_im = resize_image(final_size, im)
                new_im.save(f'cleaned_images/{item}')
    except FileExistsError:
        pass


def resize_image(final_size, im: Image):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def flatten_image():
    dirs = os.listdir('cleaned_images')
    arr = np.empty((0,786432), int)
    for item in dirs:
        im = Image.open(f'cleaned_images/{item}')
        np_img = np.array(im)
        row = np.reshape(np_img, (1,1,-1))[0]
        arr = np.append(arr, row, axis=0)
    return arr
