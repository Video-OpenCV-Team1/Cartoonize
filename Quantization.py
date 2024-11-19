import numpy as np
import cv2
import random

def color_quantization(image, human_data, max_iterations, k):
    rows, cols, _ = image.shape
    result_image = np.zeros_like(image, dtype=np.uint8)

    # Extract unique labels from human_data
    unique_labels = np.unique(human_data)
    unique_labels = unique_labels[unique_labels > 0]  # Ignore label 0

    for label in unique_labels:
        print(f"Processing Human {label}...")
        # Mask to extract pixels corresponding to the current label
        label_mask = human_data == label
        label_pixels = image[label_mask]  # Extract pixels for the current label

        # Perform K-means clustering for the current label
        centroids, memberships = kmeans(label_pixels, max_iterations, k)

        # Map the quantized colors back to their original positions
        quantized_pixels = np.array([centroids[cluster] for cluster in memberships], dtype=np.uint8)
        result_image[label_mask] = quantized_pixels

    print("Quantization Complete! Image reduced to multiple labels and colors.")
    return result_image

def kmeans(dataset, max_iterations, k):
    dataset = dataset.astype(float)  # Ensure data is float for calculations
    centroids = dataset[random.sample(range(len(dataset)), k)]
    print("Data Initialization Complete")

    for i in range(max_iterations):
        print(f"Iteration {i + 1}/{max_iterations}...")
        # Compute distances and assign memberships
        distances = np.linalg.norm(dataset[:, np.newaxis] - centroids, axis=2)
        memberships = np.argmin(distances, axis=1)
        for cluster in range(k):
            points_in_cluster = dataset[memberships == cluster]
            if len(points_in_cluster) > 0:
                centroids[cluster] = points_in_cluster.mean(axis=0)

    print("KMeans Complete")
    return centroids, memberships

# Load an image
image = cv2.imread("./rose.jpg")
human_data = np.random.randint(0, 3, size=image.shape[:2])  # Random labels for testing

cv2.imshow("Original Image", image)
cv2.imshow("Human Data", (human_data * 127).astype(np.uint8))  # Visualize human_data

# Perform color quantization
result_image = color_quantization(image, human_data, 10, 7)
cv2.imshow("Quantized Image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
