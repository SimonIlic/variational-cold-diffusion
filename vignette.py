import numpy as np
from PIL import Image

def create_circle_image(image_size, circle_radius):
    # Create a white image with the specified size
    image = np.ones((image_size, image_size), dtype=np.uint8)

    # Calculate the center coordinates of the circle
    center = (image_size // 2, image_size // 2)

    # Draw the circle on the image
    for i in range(image_size):
        for j in range(image_size):
            if np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) < circle_radius:
                image[i, j] = 0

    return image

import matplotlib.pyplot as plt

# Create the circle image
circle_image = create_circle_image(64, 0)

# Plot the image
plt.imshow(circle_image, cmap='gray')
plt.axis('off')
plt.show()
