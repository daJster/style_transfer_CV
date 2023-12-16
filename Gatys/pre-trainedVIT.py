from vit_keras import vit
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as L
import keras
import numpy as np
import pathlib
import tensorflow_addons as tfa

# set as training data
train_path = '../dataset'

dataset = keras.utils.image_dataset_from_directory(
    train_path,
    batch_size=32,
    image_size=(1024, 1024),
    shuffle=True)

image_size = 1024

vit_model = vit.vit_b16(
        image_size = image_size,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 5)

class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
plt.figure(figsize=(4, 4))
batch_size = 16
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2

print(dataset)
# x = dataset.next()
# print('x', x)
for images, labels in dataset.take(1):
    plt.imshow(images[0].numpy().astype("uint8"))
    image = images[0].numpy()
    plt.imshow(image.astype('uint8'))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size = (image_size, image_size)
)

patches = Patches(patch_size)(resized_image)
print(f'Image size: {image_size} X {image_size}')
print(f'Patch size: {patch_size} X {patch_size}')
print(f'Patches per image: {patches.shape[1]}')
print(f'Elements per patch: {patches.shape[-1]}')

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))

for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype('uint8'))
    plt.axis('off')

plt.tight_layout()
plt.show()


model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(3, 'softmax')
    ],
    name = 'vision_transformer')

model.summary()
