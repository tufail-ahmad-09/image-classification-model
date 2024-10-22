from PIL import Image
import numpy as np
import os

# Set your input directories for Real and Fake images
input_dirs = ['./data/Real', './data/Fake']  # Adjust these paths as needed
target_size = (64, 64)  # Adjust this size as needed
all_images = []

# Resize images in both directories
for input_dir in input_dirs:
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
            img = Image.open(os.path.join(input_dir, filename))
            img = img.resize(target_size, Image.LANCZOS)  # Resize image
            all_images.append(np.array(img))  # Convert to array and add to the list

# Convert list of images to numpy array
images_array = np.array(all_images)

# Save the resized images as a .npy file
np.save('./data/resized_images.npy', images_array)

print(f'Saved resized images to data/resized_images.npy with shape {images_array.shape}')
