from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from PIL import Image


# Prepare an image as an input for the network
def process_array_image(arr_image, shape):
    image = Image.fromarray(arr_image, 'RGB')
    return process_pil_image(image, shape)


def process_pil_image(image, shape):
    # save the original image shape
    width, height = image.size
    # reshape the image with the required size
    image = image.resize(shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height
