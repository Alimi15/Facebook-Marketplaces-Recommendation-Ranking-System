from PIL import Image
from torchvision.transforms import ToTensor
import clean_images

class Image_Processor():

    def process_image(img: Image):
        transform = ToTensor()
        img = transform(img)
        img = img[None]
        return img

    def clean_image(im: Image):
        new_im = clean_images.resize_image(512, im)
        return new_im