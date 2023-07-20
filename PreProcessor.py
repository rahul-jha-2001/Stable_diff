import cv2
import numpy as np
from PIL import Image

image = Image.open('D:\stable Diffusion\Stable_diff\photo_2023-04-08_13-47-00 (2).jpg')
image = np.array(image)

low_threshold = 100
high_threshold = 200
def Canny(img:np.array,low_threshold,high_threshold):

    img = cv2.Canny(img,low_threshold,high_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    canny_image = Image.fromarray(img)
    return(canny_image)
image = Canny(image,high_threshold,low_threshold)
image.show()