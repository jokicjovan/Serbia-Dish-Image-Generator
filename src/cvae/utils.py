import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def train_step(data, model, optimizer):
    with tf.GradientTape() as tape:
        model_output = model(data, is_train=True)

    trainable_variables = model.trainable_variables
    grads = tape.gradient(model_output['loss'], trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

    total_loss = model_output['loss'].numpy().mean()
    recon_loss = model_output['reconstr_loss'].numpy().mean()
    latent_loss = model_output['latent_loss'].numpy().mean()

    return total_loss, recon_loss, latent_loss

def convert_batch_to_image_grid(image_batch, dim=None):
    """
    Convert a batch of images to a grid for visualization.
    Automatically determines grid layout based on batch size.

    Args:
        image_batch: numpy array of shape (batch_size, dim, dim, 3)
        dim: image dimension (default 64)

    Returns:
        Grid image
    """
    batch_size = image_batch.shape[0]
    if dim is None:
        dim = image_batch.shape[1]

    # Determine grid dimensions
    if batch_size == 32:
        rows, cols = 4, 8
    elif batch_size == 16:
        rows, cols = 4, 4
    elif batch_size == 64:
        rows, cols = 8, 8
    elif batch_size == 8:
        rows, cols = 2, 4
    elif batch_size == 4:
        rows, cols = 2, 2
    elif batch_size == 1:
        return image_batch[0]
    else:
        # Find a reasonable grid layout
        rows = int(np.sqrt(batch_size))
        while batch_size % rows != 0:
            rows -= 1
        cols = batch_size // rows

    # Pad if necessary to fill the grid
    grid_size = rows * cols
    if batch_size < grid_size:
        padding = np.zeros((grid_size - batch_size, dim, dim, 3), dtype=image_batch.dtype)
        image_batch = np.concatenate([image_batch, padding], axis=0)
    elif batch_size > grid_size:
        image_batch = image_batch[:grid_size]

    # Reshape to grid
    reshaped = (image_batch.reshape(rows, cols, dim, dim, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows * dim, cols * dim, 3))

    return reshaped


def save_data(file_name, data):
    with open((file_name + '.pickle'), 'wb') as openfile:
        pickle.dump(data, openfile)
    print(f"Data saved to: {file_name}.pickle")


def read_data(file_name):
    with open((file_name + '.pickle'), "rb") as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
            except EOFError:
                break
    return objects


def plot_losses(losses_dict, save_path=None):
    epochs = range(1, len(losses_dict['total']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    axes[0].plot(epochs, losses_dict['total'], 'b-', linewidth=2)
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, losses_dict['reconstruction'], 'g-', linewidth=2)
    axes[1].set_title('Reconstruction Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)

    # Latent loss
    axes[2].plot(epochs, losses_dict['latent'], 'r-', linewidth=2)
    axes[2].set_title('Latent Loss (KL Divergence)', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")

    plt.show()
    plt.close()