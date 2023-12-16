import os

# global variables
LEARNING_RATE = 4.0
STYLE_WEIGHT = 1.0
CONTENT_WEIGHT = 0.025
TOTAL_VARIATION_WEIGHT = 8.5e-05
CHECK_ITER = 200
CHECK_LOSS = 4
N_ITER = 1200
RESIZE_HEIGHT = 607
MERGE_METHOD = 'mean'
APPLY_COLOR_PRESERVATION = False
APPLY_CORRELATION_CHAIN = False

# The layer to use for the content loss.
CONTENT_LAYER_NAME = "block5_conv2"
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
PERCEPTUAL_LAYER_NAMES = [
    "block3_conv3",
    "block4_conv3",
]

if APPLY_CORRELATION_CHAIN:
    STYLE_LAYER_NAMES = [
        "block1_conv1", "block1_conv2",
        "block2_conv1", "block2_conv2",
        "block3_conv1", "block3_conv2", "block3_conv3", "block3_conv4",
        "block4_conv1", "block4_conv2", "block4_conv3", "block4_conv4",
        "block5_conv1", "block5_conv2", "block5_conv3", "block5_conv4",
    ]


    
def print_param():
    print("Parameters:")
    print("\tNumber of iterations:", N_ITER)
    print("\tLearning rate:", LEARNING_RATE)
    print("\tStyle weight:", STYLE_WEIGHT)
    print("\tContent weight:", CONTENT_WEIGHT)
    print("\tTotal variation weight:", TOTAL_VARIATION_WEIGHT)
    print("\tCheck-iter:", CHECK_ITER)
    print("\tStyle layer names:", STYLE_LAYER_NAMES)
    print("\tPerceptual layer names:", PERCEPTUAL_LAYER_NAMES)
    print("\tContent layer name:", CONTENT_LAYER_NAME)
    

def filter_files_by_names(directory_path, names_style):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    matching_files = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(tuple(valid_extensions)):
            for style_name in names_style:
                if filename.lower().startswith(style_name.lower()):
                    matching_files.append(os.path.join(directory_path, filename))
                    break

    return matching_files