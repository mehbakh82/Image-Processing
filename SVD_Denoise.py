import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def denoise_image(image_path, rank):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    U, s, V = np.linalg.svd(img_array, full_matrices=False)

    s_denoised = s.copy()
    s_denoised[rank:] = 0

    denoised_img_array = U @ np.diag(s_denoised) @ V
    denoised_img_array = np.clip(denoised_img_array, 0, 255).astype(np.uint8)
    denoised_img = Image.fromarray(denoised_img_array, 'L')

    return denoised_img


ranks = [10, 30, 50, 100]

image_path = '/Users/mehbakh82/PycharmProjects/LA_Project_Q2/noisy.jpg'

for rank in ranks:
    denoised_img = denoise_image(image_path, rank)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(image_path).convert('L'), cmap='gray')
    axs[0].set_title('Noisy Image')
    axs[0].axis('off')

    axs[1].imshow(denoised_img, cmap='gray')
    axs[1].set_title('Denoised Image (Rank: {})'.format(rank))
    axs[1].axis('off')

    plt.show()
