import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2

def read_image(image_path):
    image = cv2.imread(image_path)
    assert image is not None, f"Error: Unable to load image at {image_path}"

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def gmm(img_rgb, n_components):
    pixels = img_rgb.reshape(-1, 3)

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(pixels)

    # Get the predicted labels for each pixel
    labels = gmm.predict(pixels)

    # Assign the mean color to each pixel based on the predicted label
    segmented_pixels = gmm.means_[labels]

    # Reshape the segmented pixels to the original image shape
    segmented_img = segmented_pixels.reshape(img_rgb.shape)

    # Ensure pixel values are within the valid range
    segmented_img = np.clip(segmented_img, 0, 255).astype(np.uint8)

    return segmented_img

def plt_imgs(*images):
    titles = [
        "Original Image", "Clustered Image (K=2)", "Clustered Image (K=3)",
        "Clustered Image (K=4)", "Clustered Image (K=5)", "Clustered Image (K=6)"
    ]

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        if i != 0:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    image_path = "/Users/aj/UCF_Courses/Spring_2025/STA4724/hw/HW3/image.jpg"
    image_rgb = read_image(image_path)
    
    gmm_2 = gmm(img_rgb=image_rgb, n_components=2)
    gmm_3 = gmm(img_rgb=image_rgb, n_components=3)
    gmm_4 = gmm(img_rgb=image_rgb, n_components=4)
    gmm_5 = gmm(img_rgb=image_rgb, n_components=5)
    gmm_6 = gmm(img_rgb=image_rgb, n_components=6)

    plt_imgs(image_rgb, gmm_2, gmm_3, gmm_4, gmm_5, gmm_6)

if __name__ == "__main__":
    main()
