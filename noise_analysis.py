import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path to the noisy label file
label_path = r'C:\Users\Satvik\Documents\dev\Noise Estimation\dataset\training_noisy_labels\0_9_2_418_209.png'

# Load the PNG label file
label_image = Image.open(label_path)

# Convert image to numpy array
label_array = np.array(label_image)

# Plot the label image using a grayscale colormap
plt.imshow(label_array, cmap='gray')
plt.colorbar()
plt.title("Noisy Label Visualization")
plt.show()

# Check unique pixel values
unique_values = np.unique(label_array)
print("Unique pixel values in the label:", unique_values)


