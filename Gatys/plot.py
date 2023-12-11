import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np    
import os
import imageio


def plot_losses(losses, learning_rate):
    """
    Plot the losses over time on the same plot.

    Parameters:
    - losses (dict): A dictionary of loss values over time.
    - learning_rate (float): Learning rate used in the training.
    """
    df = pd.DataFrame(losses)
    if 'Epoch' not in df.columns:
        df['Epoch'] = range(1, len(df) + 1)
        
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    for loss_type, loss_values in losses.items():
        sns.lineplot(x='Epoch', y=loss_type, data=df, label=loss_type)


    plt.title(f'Weighted losses Over Time, lr: {learning_rate}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./plots/loss_evo_gatys_{learning_rate}.png")
    plt.close()


def plot_features(tensor, layer_name):
    """
    Plot a square of monochromatic images from a TensorFlow tensor.

    Parameters:
    - tensor (Tensor): TensorFlow tensor of shape (1, x, y, z).
    """
    _, x, y, z = tensor.shape

    # Calculate the number of rows and columns for the subplot
    subplot_rows = subplot_cols = int(np.sqrt(z))

    # Create a subplot for the images
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(8, 8))

    # Reshape the tensor to (x, y, z)
    reshaped_tensor = np.squeeze(tensor, axis=0)

    # Plot each monochromatic image
    for i in range(z):
        row = i // subplot_cols
        col = i % subplot_cols
        image = reshaped_tensor[:, :, i]  # Select every second channel
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        
    title = f"Features of Layer {layer_name}"
    plt.suptitle(title, fontsize=16, y=0.95)

    plt.savefig(f"./plots/features_evo_gatys_{layer_name}.png")
    plt.close()
    
    

def create_gif(input_folder, output_gif_dir):
    """
    Create a GIF from multiple images in a folder.

    Parameters:
    - input_folder (str): Path to the folder containing images.
    - output_gif (str): Path to the output GIF file.
    """
    images = []
    
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the image files to ensure correct order
    image_files.sort()

    # Read each image and append it to the 'images' list
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        images.append(imageio.v2.imread(image_path))

    # Create the GIF
    imageio.mimsave(f"{output_gif_dir}result_{len(image_files)}.gif", images, duration=0.1)  # Adjust the duration as needed
    print(f"gif created.")


if __name__ == "__main__" :
    create_gif('./result/', './plots/')