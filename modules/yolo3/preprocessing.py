from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from PIL import Image


# Prepare an image as an input for the network
def process_image(raw_image, shape):
    image = Image.fromarray(raw_image, 'RGB')
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
