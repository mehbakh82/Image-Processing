import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def compress_image(image_path, compression_rate):
    img = Image.open(image_path).convert('L')
    original_shape = img.size[::-1]
    img_array = np.array(img)

    U, s, V = np.linalg.svd(img_array)

    k = int(np.ceil(compression_rate * min(img_array.shape)))

    U_comp = U[:, :k]
    s_comp = s[:k]
    V_comp = V[:k, :]

    compressed_img_array = U_comp @ np.diag(s_comp) @ V_comp
    compressed_img_array = np.clip(compressed_img_array, 0, 255).astype(np.uint8)
    compressed_img = Image.fromarray(compressed_img_array, 'L')

    return compressed_img, original_shape


compression_rates = [0.001, 0.005, 0.01, 0.05, 0.1]

image_path = '/Users/mehbakh82/PycharmProjects/LA_Project_Q2/dog.jpg'

for rate in compression_rates:
    compressed_img, original_shape = compress_image(image_path, rate)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(image_path).convert('L'), cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(compressed_img, cmap='gray')
    axs[1].set_title('Compressed Image (Rate: {})'.format(rate))
    axs[1].axis('off')

    plt.show()