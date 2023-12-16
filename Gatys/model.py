from tensorflow import keras
from keras.applications import vgg19


def get_model():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights = 'imagenet', include_top = False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in VGG19 (as a dict).
    return keras.Model(inputs = model.inputs, outputs = outputs_dict)

def get_optimizer(learning_rate):
    return keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = learning_rate, decay_steps = 445, decay_rate = 0.98
        )
    )