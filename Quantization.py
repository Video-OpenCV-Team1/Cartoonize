import numpy as np
import cv2
import random

def color_quantization(image, max_iteration, k):
    centroids, memberships = kmeans(image, max_iteration, k)
    rows, cols, _ = image.shape
    quantized_image = np.array([centroids[cluster] for cluster in memberships], dtype=np.uint8)
    quantized_image = quantized_image.reshape(rows, cols, -1)
    print(f"Quantization Complete! Image reduced to {k} colors.")
    return quantized_image

def kmeans(image, max_iterations, k):
    dataset = image.reshape((-1, 3)).astype(float)
    centroids = dataset[random.sample(range(len(dataset)), k)]
    print("Data Initialization Complete")

    for i in range(max_iterations):
        print(f"Iteration {i + 1}/{max_iterations}...")
        distances = np.linalg.norm(dataset[:, np.newaxis] - centroids, axis=2)
        memberships = np.argmin(distances, axis=1)
        for cluster in range(k):
            points_in_cluster = dataset[memberships == cluster]
            if len(points_in_cluster) > 0:
                centroids[cluster] = points_in_cluster.mean(axis=0)
    print("KMeans Complete")
    return centroids, memberships

image = cv2.imread("./rose.jpg")
cv2.imshow("Original Image", image)
result_image = color_quantization(image, 10, 7)
cv2.imshow("Quantized Image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()