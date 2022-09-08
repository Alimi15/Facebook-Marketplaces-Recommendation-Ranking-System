from PIL import Image
import os

def clean_image_data(filepath):
    try:
        os.mkdir("cleaned_images")
    except FileExistsError:
        pass
    dirs = os.listdir(filepath)
    final_size = 512
    for n, item in enumerate(dirs, 1):
        im = Image.open(os.path.join(filepath, item))
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{n}_resized.jpg')


def resize_image(final_size, im: Image):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im