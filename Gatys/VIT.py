import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
print('tf version', tf.__version__)
print('keras', keras.__version__)
#from tensorflow.keras.applications
from tensorflow.python.keras import ops
from tensorflow.python.keras import layers
import time
from main1 import preprocess_image, get_result_image_size, get_optimizer, get_model, compute_loss, save_result

f = open('vit_loss.txt', 'w')
# Generated image size
RESIZE_HEIGHT = 607

NUM_ITER = 1000

# Weights of the different loss components
CONTENT_WEIGHT = 8e-4 
STYLE_WEIGHT = 8e-1

# hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size # TO CHANGE
patch_size = 6  # Size of the patches to be extract from the input images # TO CHANGE
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier

# implement MLP
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# implement patch creations as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
    
# Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

# build the VIT model    
def create_vit_classifier():
    inputs = keras.Input(shape=RESIZE_HEIGHT)
    # Augment data.
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# TO CHANGE
CONTENT_LAYER_NAME = "block5_conv2" 
# List of layers to use for the style loss.
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

# apply style transfer using VIT architecture

# main part
if __name__ == "__main__":
    # Prepare content, style images
    content_image_path = './dataset/paris.jpg'
    style_image_path = './dataset/starry_night.jpg'
    result_height, result_width = get_result_image_size(content_image_path, RESIZE_HEIGHT)
    print("result resolution: (%d, %d)" % (result_height, result_width))

    # Preprocessing
    content_tensor = preprocess_image(content_image_path, result_height, result_width)
    style_tensor = preprocess_image(style_image_path, result_height, result_width)
    # generated_image = tf.Variable(tf.random.uniform(content_tensor.shape, dtype=tf.dtypes.float32)) # gaussian noise
    generated_image = tf.Variable(preprocess_image(content_image_path, result_height, result_width)) # content image

    # Build model
    model = create_vit_classifier()
    optimizer = get_optimizer()
    print(model.summary())

    f.write(str(optimizer.get_config()))

    content_features = model(content_tensor)
    style_features = model(style_tensor)
    
    print('before merging styles')
    
    # merge style features with mean
    # style_features = merge_style_features(style_features1, style_features2)
    
    start_time = time.time()
    # Optimize result image
    for iter in range(NUM_ITER):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, generated_image, content_features, style_features)

        grads = tape.gradient(loss, generated_image)

        print("iter: %4d, loss: %8.f" % (iter, loss))
        f.write("iter: %4d, loss: %8.f\n" % (iter, loss))
        optimizer.apply_gradients([(grads, generated_image)])

        if (iter + 1) % 200 == 0:
            assert(os.path.isdir('result'))
            name = "result/generated_at_iteration_%d.png" % (iter + 1)
            save_result(generated_image, result_height, result_width, name)
            time_for_200_it = time.time() - start_time
            print('time_for_200_iterations: ', time_for_200_it)

    name = "result/result_%d_%f_%f.png" % (NUM_ITER, CONTENT_WEIGHT, STYLE_WEIGHT)
    save_result(generated_image, result_height, result_width, name)
    final_time = time.time() - start_time
    print('generation of image lasted : ', final_time, 's')
    f.close()