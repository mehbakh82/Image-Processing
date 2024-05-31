import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def denoise_image(image_path, radius):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    img_fft = np.fft.fftshift(np.fft.fft2(img_array))

    center = np.array(img_fft.shape) // 2
    y, x = np.ogrid[:img_fft.shape[0], :img_fft.shape[1]]
    mask = np.sqrt((x - center[1])**2 + (y - center[0])**2) < radius
    img_fft = img_fft * mask

    denoised_img_array = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft))).astype(np.uint8)
    denoised_img = Image.fromarray(denoised_img_array, 'L')

    return denoised_img


radii = [10, 30, 50, 100]

image_path = '/Users/mehbakh82/PycharmProjects/LA_Project_Q2/noisy.jpg'

for radius in radii:
    denoised_img = denoise_image(image_path, radius)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(image_path).convert('L'), cmap='gray')
    axs[0].set_title('Noisy Image')
    axs[0].axis('off')

    axs[1].imshow(denoised_img, cmap='gray')
    axs[1].set_title('Denoised Image (Radius: {})'.format(radius))
    axs[1].axis('off')

    plt.show()
