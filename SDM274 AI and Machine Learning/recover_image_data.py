import numpy as np
import matplotlib.pyplot as plt

def recover_image_data(encoded_data, image_size=(28, 28)):
    height, width = image_size
    image = np.zeros(image_size, dtype=np.uint8)

    if encoded_data.ndim > 1:
        encoded_data = encoded_data.flatten()

    for row in range(height):
        base = row * 4
        if base + 3 < len(encoded_data):
            a, b, c, d = encoded_data[base:base+4]

            if a != -1 and b != -1:
                for col in range(a, b + 1):
                    if 0 <= col < width:
                        image[row, col] = 1
            if c != -1 and d != -1:
                for col in range(c, d + 1):
                    if 0 <= col < width:
                        image[row, col] = 1
    return image