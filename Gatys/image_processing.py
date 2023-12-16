from tensorflow import keras
from keras.applications import vgg19
import tensorflow as tf
from skimage import exposure
import numpy as np
from PIL import Image

def get_result_image_size(image_path, result_height):
    image_width, image_height = keras.preprocessing.image.load_img(image_path).size
    print('width and height', image_width, image_height)
    result_width = int(image_width * result_height / image_height)
    return result_height, result_width

def preprocess_image(image_path, target_height, target_width):
    img = keras.preprocessing.image.load_img(image_path, target_size = (target_height, target_width))
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis = 0)
    arr = vgg19.preprocess_input(arr)
    return tf.convert_to_tensor(arr)

def match_histograms(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    """
    source_hist, bin_centers = exposure.histogram(source)
    template_hist, _ = exposure.histogram(template)
    matched = exposure.match_histograms(source, template, multichannel=True)
    return matched

def preprocess_image_with_color_matching(content_image_path, style_image_path, target_height, target_width):
    content_img = keras.preprocessing.image.load_img(content_image_path, target_size=(target_height, target_width))
    style_img = keras.preprocessing.image.load_img(style_image_path, target_size=(target_height, target_width))

    content_arr = keras.preprocessing.image.img_to_array(content_img)
    style_arr = keras.preprocessing.image.img_to_array(style_img)

    # histogram matching
    matched_style_arr = match_histograms(style_arr, content_arr)

    matched_style_arr = np.expand_dims(matched_style_arr, axis=0)
    matched_style_arr = vgg19.preprocess_input(matched_style_arr)

    return tf.convert_to_tensor(matched_style_arr)

def original_color_transform(content, generated, mask=None):
    generated_resized = Image.fromarray(generated, mode='RGB').resize(content.shape[1::-1], Image.LANCZOS)

    # convert images to YCbCr color space
    content_yCbCr = Image.fromarray(content, mode='RGB').convert('YCbCr')
    generated_yCbCr = generated_resized.convert('YCbCr')

    content_np = np.array(content_yCbCr)
    generated_np = np.array(generated_yCbCr)

    if mask is None:
        generated_np[:, :, 1:] = content_np[:, :, 1:]  # replacing CbCr of generated with the content's
    else:
        width, height, _ = generated_np.shape
        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    generated_np[i, j, 1:] = content_np[i, j, 1:]

    # convert back to RGB color space
    generated_rgb = Image.fromarray(generated_np, mode='YCbCr').convert('RGB')
    return np.array(generated_rgb)

# Util function to convert a tensor into a valid image
def deprocess_image(tensor, result_height, result_width):
    tensor = tensor.numpy()
    tensor = tensor.reshape((result_height, result_width, 3))

    # Remove zero-center by mean pixel
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.680

    # 'BGR'->'RGB'
    tensor = tensor[:, :, ::-1]
    return np.clip(tensor, 0, 255).astype("uint8")
