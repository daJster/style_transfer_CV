import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import vgg19
import time
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image


f = open("loss.txt", "w")

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU available. Please install GPU version of TensorFlow.")

# Generated image size
RESIZE_HEIGHT = 607

NUM_ITER = 3000

# Weights of the different loss components
CONTENT_WEIGHT = 8e-04 # 8e-4 # test different values
STYLE_WEIGHT = 1 # 8e-1, 8e-4 # test different values
TOTAL_VARIATION_WEIGHT = 1e-05   #8.5e-05

# The layer to use for the content loss.
CONTENT_LAYER_NAME = "block5_conv2" #  initial : "block5_conv2" # test different values # block5_conv3, block5_conv4, block5_conv5

# List of layers to use for the style loss.
# STYLE_LAYER_NAMES = [
#     "block1_conv1", "block1_conv2",
#     "block2_conv1", "block2_conv2",
#     "block3_conv1", "block3_conv2", "block3_conv3", "block3_conv4",
#     "block4_conv1", "block4_conv2", "block4_conv3", "block4_conv4",
#     "block5_conv1", "block5_conv2", "block5_conv3", "block5_conv4",
# ]

STYLE_LAYER_NAMES = [
    "block1_conv1", 
    "block2_conv1",
    "block3_conv1",
    "block4_conv1"
    "block5_conv1",
]




PERCEPTUAL_LAYER_NAMES = [
    "block3_conv3",
    "block4_conv3",
]






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

def get_model():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights = 'imagenet', include_top = False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in VGG19 (as a dict).
    return keras.Model(inputs = model.inputs, outputs = outputs_dict)

def get_optimizer():
    return keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 16.0, decay_steps = 445, decay_rate = 0.98 # test different values
            #initial_learning_rate = 4.0, decay_steps = 376, decay_rate = 0.98
        )
    )






def compute_loss(feature_extractor, combination_image, content_features, style_features):
    combination_features = feature_extractor(combination_image)
    loss_content = compute_content_loss(content_features, combination_features)
    loss_style = compute_style_loss(style_features, combination_features, combination_image.shape[1] * combination_image.shape[2])
    loss_total_variation = total_variation_loss(combination_image)

    return (CONTENT_WEIGHT * loss_content +
            STYLE_WEIGHT * loss_style +
            TOTAL_VARIATION_WEIGHT * loss_total_variation)


# A loss function designed to maintain the 'content' of the original_image in the generated_image
def compute_content_loss(content_features, combination_features):
    original_image = content_features[CONTENT_LAYER_NAME]
    generated_image = combination_features[CONTENT_LAYER_NAME]

    return tf.reduce_sum(tf.square(generated_image - original_image)) / 2


def compute_style_loss(style_features, combination_features, combination_size):
    loss_style = 0
    num_style_layers = len(STYLE_LAYER_NAMES) - 1  # Adjust for layer pairs

    for i in range(num_style_layers):
        # Current and next layer names
        current_layer_name = STYLE_LAYER_NAMES[i]
        next_layer_name = STYLE_LAYER_NAMES[i + 1]

        # Style features for current and next layers
        current_style_feature = style_features[current_layer_name][0]
        next_style_feature = style_features[next_layer_name][0]

        # Combination features for current and next layers
        current_combination_feature = combination_features[current_layer_name][0]
        next_combination_feature = combination_features[next_layer_name][0]

        # Compute style loss for current and next layers
        sl1 = style_loss(current_style_feature, current_combination_feature, combination_size)
        sl2 = style_loss(next_style_feature, next_combination_feature, combination_size)

        # Differential style loss
        sl = sl1 - sl2

        # Apply geometric weighting
        layer_weight = 1 / (2 ** (num_style_layers - i))
        loss_style += layer_weight * sl

    # Normalize the style loss
    return loss_style / num_style_layers







# The "style loss" is designed to maintain the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of feature maps from the style reference image and from the generated image
def style_loss(style_features, combination_features, combination_size):
    S = gram_matrix(style_features)
    C = gram_matrix(combination_features)
    channels = style_features.shape[2]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (combination_size ** 2))

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    # features -= 1  # Activation shift
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def total_variation_loss(image):
    a = tf.square(image[:, :RESIZE_HEIGHT - 1, :result_width - 1, :] - image[:, 1:, :result_width - 1, :])
    b = tf.square(image[:, :RESIZE_HEIGHT - 1, :result_width - 1, :] - image[:, :RESIZE_HEIGHT - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))







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
    generated_resized = Image.fromarray(generated, mode='RGB').resize(content.shape[1::-1], Image.Resampling.LANCZOS)

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










def save_result(generated_image, result_height, result_width, name):
    img = deprocess_image(generated_image, result_height, result_width)
    keras.preprocessing.image.save_img(name, img)

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



def check_get_shape(style_features_list) :
    if not style_features_list :
        raise ValueError("style_features_list must contain at least one style feature.")

    shapes = [style_feature.shape for style_feature in style_features_list]
    if len(set(shapes)) != 1 :
        raise ValueError(f"All style features must have the same shape. Found shapes: {shapes}")

    return shapes[0]

# function to merge style features using mean
def merge_style_features_mean(style_features_list) :

    merged_style_features = {}

    for key in STYLE_LAYER_NAMES :
        i = 0
        for style_feature in style_features_list :
            if i == 0 :
                merged_style_features[key] = style_feature[key]
                i += 1
            else :
                merged_style_features[key] += style_feature[key]

        merged_style_features[key] /= len(style_features_list) # mean
    return merged_style_features

def merge_style_features_percentage(style_features_list, percentages) :

    merged_style_features = {}
    for layer in STYLE_LAYER_NAMES :
        merged_style_features[layer] = np.zeros(style_features_list[0][layer].shape)

        for z in range(style_features_list[0][layer].shape[3]):

            selected_style = np.random.multinomial(1, percentages) # multinomial distribution
            index = np.argmax(selected_style)
            merged_style_features[layer][:, :, :, z] = style_features_list[index][layer][:, :, :, z]

        merged_style_features[layer] = tf.convert_to_tensor(merged_style_features[layer], dtype=tf.float32)

    return merged_style_features


def merge_style_features_alternate(style_features_list) :
    '''
    alternate style features
    '''
    n = len(style_features_list)
    merged_style_features = {}
    for layer in STYLE_LAYER_NAMES :
        merged_style_features[layer] = np.zeros(style_features_list[0][layer].shape)

        for z in range(style_features_list[0][layer].shape[3]):

            merged_style_features[layer][:, :, :, z] = style_features_list[z%n][layer][:, :, :, z] # modulo

        merged_style_features[layer] = tf.convert_to_tensor(merged_style_features[layer], dtype=tf.float32)

    return merged_style_features



def display_style_features(style_features_list):
    for key in style_features_list[0].keys():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed
        for i in range(3):
            axes[i].imshow(style_features_list[0][key][0, :, :, i], cmap='gray')  # Assuming grayscale images, adjust cmap if needed
            axes[i].axis('off')  # Turn off axis labels for clarity
        fig.title(key)
        plt.show()

'''
def merge_style_features_percentage(style_features1, style_features2, percentage):
    merged_features = {}
    for key in style_features1.keys():
        # Ensure the shapes of the arrays are the same
        if style_features1[key].shape == style_features2[key].shape:
            # Randomly choose indices to replace based on the specified percentage
            mask = np.random.rand(*style_features1[key].shape) < percentage

            # Replace values from style_features1 with values from style_features2 based on the mask
            merged_features[key] = np.where(mask, style_features2[key], style_features1[key])
        else:
            # If shapes are different, use the values from style_features1 directly
            merged_features[key] = style_features1[key]

    return merged_features
'''

if __name__ == "__main__":
    # Prepare content, style images
    content_image_path = './dataset/einstein.jpg'
    style_image_path_1 = './dataset/manga/manga-2.jpg'
    #style_image_path_2 = './dataset/starry_night.jpg'
    result_height, result_width = get_result_image_size(content_image_path, RESIZE_HEIGHT)
    print("result resolution: (%d, %d)" % (result_height, result_width))

    # Preprocessing
    content_tensor = preprocess_image(content_image_path, result_height, result_width)

    # without color preservation
    style_tensor1 = preprocess_image(style_image_path_1, result_height, result_width)

    # color preservation (color matching)
    # style_tensor1 = preprocess_image_with_color_matching(content_image_path, style_image_path_2, result_height, result_width)

    # content image rgb for color preservation
    content_image_rgb = np.array(Image.open(content_image_path))


    # style_tensor2 = preprocess_image(style_image_path_2, result_height, result_width)
    # generated_image = tf.Variable(tf.random.uniform(content_tensor.shape, dtype=tf.dtypes.float32)) # gaussian noise
    generated_image = tf.Variable(preprocess_image(content_image_path, result_height, result_width)) # content image

    # Build model
    model = get_model()

    optimizer = get_optimizer()
    print(model.summary())

    f.write(str(optimizer.get_config()))

    content_features = model(content_tensor)
    # style_features_list = [model(style_tensor1), model(style_tensor2)]
    style_features_list = [model(style_tensor1)]
    print('before merging styles')
    # method 1
    # style_features = merge_style_features_mean(style_features_list) # ok
    # method 2
    # style_features = merge_style_features_percentage(style_features_list, [0.5, 0.5])
    # style_features = merge_style_features_alternate(style_features_list)
    style_features1 = model(style_tensor1)
    # style_features2 = model(style_tensor2)
    # print(style_features2)
    # print(list(style_features1.values())[0])
    # print(type(list(style_features2.values())[0]))


    # merge style features with mean
    # style_features = merge_style_features(style_features1, style_features2)
    # style_features = merge_style_features_percentage(style_features1, style_features2,5)
    print('styles merged')

    apply_color_preservation = True

    start_time = time.time()
    # Optimize result image
    for iter in range(NUM_ITER):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, generated_image, content_features, style_features1)

        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])

        print("iter: %4d, loss: %8.f" % (iter, loss))

        if (iter + 1) % 100 == 0:
            generated_image_rgb = deprocess_image(generated_image, result_height, result_width)

            if apply_color_preservation:
                # Apply the color preservation
                preserved_color_image = original_color_transform(content_image_rgb, generated_image_rgb)
                preserved_image_name = "result/image_at_iteration_%d.png" % (iter + 1)
                Image.fromarray(preserved_color_image).save(preserved_image_name)
            else:
                # Save without color preservation
                Image.fromarray(generated_image_rgb).save("result/image_at_iteration_%d.png" % (iter + 1))

    # Save final image after completing all iterations
    final_generated_image_rgb = deprocess_image(generated_image, result_height, result_width)
    if apply_color_preservation:
        final_preserved_color_image = original_color_transform(content_image_rgb, final_generated_image_rgb)
        Image.fromarray(final_preserved_color_image).save("result/final_result.png")
    else:
        Image.fromarray(final_generated_image_rgb).save("result/final_result.png")

    final_time = time.time() - start_time
    print('generation of image lasted : ', final_time, 's')
    f.close()

    # vision transformer on Tensorflow : https://www.tensorflow.org/api_docs/python/tfm/vision/configs/backbones/VisionTransformer