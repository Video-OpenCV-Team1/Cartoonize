import numpy as np


def ()

def kmeans(X, max_iterations, k):
    centroids = np.full((k, 2), 0, dtype=float)
    Z = np.full((len(X), k), 0, dtype=int)
    memberships = np.full(len(X), 0, dtype=int)

    for centroid in centroids :
        boundary = GetDataBound(X)
        centroid[0] = random.uniform(boundary[0][0], boundary[0][1])
        centroid[1] = random.uniform(boundary[1][0], boundary[1][1])

    for i in range(max_iterations):
        for pointIndex, point in enumerate(X):
            minIndex = 0
            minDistSqr = 1000000
            for index, centroid in enumerate(centroids):
                distSqr = (centroid[0] - point[0]) ** 2 + (centroid[1] - point[1]) ** 2
                if distSqr < minDistSqr :
                    minDistSqr = distSqr
                    minIndex = index
            for index in range(k):
                if minIndex == index:
                    Z[pointIndex][index] = 1
                else :
                    Z[pointIndex][index] = 0


        for index in range(len(centroids)):
            cnt = 0
            sum = np.array([0,0], dtype = float)
            for pointIndex, point in enumerate(X):
                if(Z[pointIndex][index] == 1) :
                    cnt += 1
                    sum += point
            if(cnt > 0):
                centroids[index] = sum / cnt

    for pointIndex in range(len(X)):
        for index, Zvalue in enumerate(Z[pointIndex]):
            if (Zvalue == 1):
                memberships[pointIndex] = index
                break
    return centroids, memberships