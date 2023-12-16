import tensorflow as tf
from param import CONTENT_WEIGHT, STYLE_WEIGHT, TOTAL_VARIATION_WEIGHT, RESIZE_HEIGHT, STYLE_LAYER_NAMES, CONTENT_LAYER_NAME, APPLY_CORRELATION_CHAIN


def compute_loss(feature_extractor, combination_image, content_features, style_features):
    combination_features = feature_extractor(combination_image)
    style_reference_features = feature_extractor(style_features)
    
    loss_content = compute_content_loss(content_features, combination_features)
    loss_total_variation = total_variation_loss(combination_image)

    if APPLY_CORRELATION_CHAIN :
        # Correlation Chain for Style Loss
        loss_style = 0
        for i in range(len(STYLE_LAYER_NAMES) - 1):
            layer_name = STYLE_LAYER_NAMES[i]
            next_layer_name = STYLE_LAYER_NAMES[i + 1]

            combination_features_current = combination_features[layer_name]
            style_reference_features_current = style_reference_features[layer_name]

            shape = tf.shape(combination_features_current).numpy()
            sl1 = style_loss(style_reference_features_current[0], combination_features_current[0], shape)

            combination_features_next = combination_features[next_layer_name]
            style_reference_features_next = style_reference_features[next_layer_name]

            shape_next = tf.shape(combination_features_next).numpy()
            sl2 = style_loss(style_reference_features_next[0], combination_features_next[0], shape_next)

            sl = sl1 - sl2
            loss_style += sl / (2 ** (len(STYLE_LAYER_NAMES) - (i + 1)))
            
    else :
        loss_style = compute_style_loss(style_features, combination_features, combination_image.shape[1] * combination_image.shape[2])


    return (CONTENT_WEIGHT * loss_content +
            STYLE_WEIGHT * loss_style +
            TOTAL_VARIATION_WEIGHT * loss_total_variation), CONTENT_WEIGHT * loss_content, STYLE_WEIGHT * loss_style, TOTAL_VARIATION_WEIGHT * loss_total_variation


# A loss function designed to maintain the 'content' of the original_image in the generated_image
def compute_content_loss(content_features, combination_features):
    original_image = content_features[CONTENT_LAYER_NAME]
    generated_image = combination_features[CONTENT_LAYER_NAME]

    return tf.reduce_sum(tf.square(generated_image - original_image)) / 2

def compute_style_loss(style_features, combination_features, combination_size):
    loss_style = 0

    for layer_name in STYLE_LAYER_NAMES:
        style_feature = style_features[layer_name][0]
        combination_feature = combination_features[layer_name][0]
        loss_style += style_loss(style_feature, combination_feature, combination_size) / len(STYLE_LAYER_NAMES)

    return loss_style

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
    if APPLY_CORRELATION_CHAIN :
        features -= 1  # Activation shift
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def total_variation_loss(image, result_width):
    a = tf.square(image[:, :RESIZE_HEIGHT - 1, :result_width - 1, :] - image[:, 1:, :result_width - 1, :])
    b = tf.square(image[:, :RESIZE_HEIGHT - 1, :result_width - 1, :] - image[:, :RESIZE_HEIGHT - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))