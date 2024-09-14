import os
import numpy as np
from PIL import Image

def compute_noise_level(label_path):
    # Load the noisy label image
    label_image = Image.open(label_path)
    
    # Convert image to numpy array
    label_array = np.array(label_image)
    
    # Compute the noise level (ratio of 1's in the label to total pixels)
    noise_level = np.sum(label_array == 1) / label_array.size
    return noise_level

def rank_images_by_noise(label_dir):
    noise_levels = []
    
    # Iterate over all label files in the directory
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        if label_file.endswith('.png'):  # Ensure we're only processing PNG files
            noise_level = compute_noise_level(label_path)
            noise_levels.append((label_file, noise_level))
    
    # Sort the images by noise level in descending order
    noise_levels.sort(key=lambda x: x[1], reverse=True)
    
    return noise_levels

# Directory containing the noisy label files
label_dir = r'C:\Users\Satvik\Documents\dev\Noise Estimation\dataset\training_noisy_labels'

# Get the ranked list of images by noise level
ranked_images = rank_images_by_noise(label_dir)

# Print the top 5 noisiest images
for image_name, noise_level in ranked_images[:5]:
    print(f"{image_name}: {noise_level:.4f} noise level")
