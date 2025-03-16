import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

def read_image(image_path):
    image = cv2.imread(image_path)
    assert image is not None, f"Error: Unable to load image at {image_path}"

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def process_img(img_rgb):
    # Reshape image to 2D array of pixels for clustering
    pixels = img_rgb.reshape(-1, 3)
    return pixels

def k_means(img, pixels, k):
    # Apply K-Means algorithm
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get segmented pixels by mapping each pixel to its cluster center
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back to the original image shape
    segmented_img = segmented_pixels.reshape(img.shape).astype(np.uint8)

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
    pixels = process_img(image_rgb)

    # Apply K-Means clustering with different K values
    kmeans_2 = k_means(image_rgb, pixels, 2)
    kmeans_3 = k_means(image_rgb, pixels, 3)
    kmeans_4 = k_means(image_rgb, pixels, 4)
    kmeans_5 = k_means(image_rgb, pixels, 5)
    kmeans_6 = k_means(image_rgb, pixels, 6)

    # Display the results
    plt_imgs(image_rgb, kmeans_2, kmeans_3, kmeans_4, kmeans_5, kmeans_6)

if __name__ == "__main__":
    main()
