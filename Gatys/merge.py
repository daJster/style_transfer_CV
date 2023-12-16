import numpy as np
import tensorflow as tf
from param import STYLE_LAYER_NAMES

def merge_style_features_mean(style_features_list):
    merged_style_features = {}

    for layer_name in STYLE_LAYER_NAMES:
        sum_value = sum(style_feature[layer_name] for style_feature in style_features_list)
        merged_style_features[layer_name] = sum_value / len(style_features_list)

    return merged_style_features

def merge_style_features_percentage(style_features_list, percentages):
    """
    Merge multiple style features based on given percentages.

    Args:
        style_features_list (list): List of style features.
        percentages (numpy.ndarray): Array of percentages.

    Returns:
        dict: Merged style features.
    """
    merged_style_features = {}
    
    for layer in STYLE_LAYER_NAMES:
        merged_style_features[layer] = np.zeros(style_features_list[0][layer].shape)
        
        for z in range(style_features_list[0][layer].shape[3]):
            selected_style = np.random.multinomial(1, percentages)
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
