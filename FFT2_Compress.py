import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def compress_image(image_path, compression_rate):
    img = Image.open(image_path).convert('L')
    original_shape = img.size[::-1]
    img_array = np.array(img)

    img_fft = np.fft.fft2(img_array)

    k = int(np.ceil(compression_rate * img_array.size))

    img_fft_flat = img_fft.flatten()

    indices = np.argsort(np.abs(img_fft_flat))
    img_fft_flat[indices[:-k]] = 0

    img_fft_comp = img_fft_flat.reshape(img_fft.shape)

    compressed_img_array = np.abs(np.fft.ifft2(img_fft_comp)).astype(np.uint8)
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


# Which change of basis seems to do better when it comes to image compression? Briefly elaborate on what makes that
# change of basis more suited for images?
"""
I think that the choice of basis depends on the specific characteristics of the image and the compression requirements 
and Both methods offer different advantages and considerations

The main advantage of SVD-based compression is its ability to capture the most significant information in the image by 
focusing on the singular values. By retaining the largest singular values and discarding the smaller ones, we preserve 
the most important image features while reducing the size of the data. This makes SVD a suitable choice for images with 
well-defined structures and localized details.

The advantage of FFT2-based compression lies in its ability to capture global frequency content, which is suitable for 
images with repetitive patterns or smooth variations. By discarding high-frequency components, which typically contain 
less perceptually important information, we can achieve compression. FFT2 is particularly useful for images with a high 
degree of regularity, such as textured or patterned images.
"""
