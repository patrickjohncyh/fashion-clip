"""
Functions required to generate attention map / heatmap based on query
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pad_to_square(image, size=224):
    original_size = image.size
    ratio = float(size) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    new_im = Image.new("RGB", size=(size, size), color=(128, 128, 128))
    new_im.paste(image.resize(new_size, Image.ANTIALIAS),
                 ((size - new_size[0]) // 2,
                  (size - new_size[1]) // 2))
    return new_im


def generate_horizontal_masks(image_size, pixel_size, channels=3):
    base_mask = np.ones((image_size, image_size, channels))
    masks = []
    n_pixels = image_size // pixel_size + 1
    for i in range(1, n_pixels):
        for j in range(i+1, n_pixels):
            m = base_mask.copy()
            m[:min(i*pixel_size, image_size), :] = 0
            m[min(j*pixel_size, image_size):, :] = 0
            masks.append(m)
    return masks

def generate_vertical_masks(image_size, pixel_size, channels=3):
    base_mask = np.ones((image_size, image_size, channels))
    masks = []
    n_pixels = image_size // pixel_size + 1
    for i in range(0, n_pixels):
        for j in range(i + 1, n_pixels):
            m = base_mask.copy()
            m[:, :min(i * pixel_size, image_size)] = 0
            m[:, min(j * pixel_size, image_size):] = 0
            masks.append(m)
    return masks


def generate_image_crops(image_path, image_size=224, pixel_size=10):

    assert pixel_size * 2 < image_size

    image = Image.open(image_path).convert("RGB")
    image = pad_to_square(image, size=image_size)
    gray = np.ones_like(image) * 128

    image_crops = [image]
    masks = [np.ones_like(image)]

    horizontal_masks = generate_horizontal_masks(image_size, pixel_size)
    vertical_masks = generate_vertical_masks(image_size, pixel_size)
    for m in horizontal_masks+vertical_masks:
        m_bar = 1-m
        m_masked = image * m + gray * m_bar
        m_masked = m_masked.astype(np.uint8)
        image_crops.append(Image.fromarray(m_masked))
        masks.append(m)
    return image_crops, masks


def display_heatmap(image, text, heatmap):
    fig = plt.figure(figsize=(15, 30), facecolor='white')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(image)
    ax2.imshow(heatmap)
    ax3.imshow(np.asarray(image) / 255. * heatmap)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.title.set_text('input image')
    ax2.title.set_text(text)

    return fig
