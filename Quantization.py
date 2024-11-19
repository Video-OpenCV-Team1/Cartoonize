import numpy as np
import cv2
import matplotlib.pyplot as plt

def quantization(image, k):
    centroids, memberships = kmeans(image, 100, k)
    rows, cols, channels = image.shape
    cnt = 0
    for row in range(rows):
        for col in range(cols):
            image[row, col] = centroids[memberships[cnt]]
            cnt += 1



def get_color_data(image):
    data = []
    rows, cols, channels = image.shape
    for row in range(rows):
        for col in range(cols):
            pixel_value = image[row, col]
            data.append(pixel_value)

    return data

def kmeans(image,max_iterations,k):
    dataset = np.array(get_color_data(image))
    centroids = np.full((k, 3), 0, dtype=float)
    Z = np.full((len(X), k), 0, dtype=int)
    memberships = np.full(len(X), 0, dtype=int)

    for centroid in centroids:
        for color_index in centroid:
            centroid[color_index] = random.randint(0, 255)

    for i in range(max_iterations):
        for point_index, point in enumerate(dataset):
            min_index = 0
            min_dist_sqr = 100000000000
            for index, centroid in enumerate(centroids):
                dist_sqr = 0
                for color_index in range(3):
                    dist_sqr += (centroid[color_index] - point[color_index]) ** 2
                if dist_sqr < min_dist_sqr:
                    min_dist_sqr = dist_sqr
                    min_index = index
            for index in range(k):
                if min_index == index:
                    Z[point_index][index] = 1
                else:
                    Z[point_index][index] = 0

        for index in range(len(centroids)):
            cnt = 0
            sum = np.array([0, 0, 0], dtype=float)
            for pointIndex, point in enumerate(X):
                if(Z[pointIndex][index] == 1) :
                    cnt += 1
                    sum += point
            if(cnt > 0):
                centroids[index] = sum / cnt

    for pointIndex in range(len(X)):
        for index, Zvalue in enumerate(Z[pointIndex]):
            if Zvalue == 1:
                memberships[pointIndex] = index
                break
    return centroids, memberships
