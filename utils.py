import numpy
from numpy import *

# calculate Ecli distance
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# randomly choose k center
def randCenter(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# do cluster
def kMeans(dataSet, k, distMeas=distEclud, createCenter=randCenter):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centriods = createCenter(dataSet, k)
    clusterChange = True
    while clusterChange:
        clusterChange = False
        for i in range(m):
            minDist = inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centriods[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True;
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centriods)
        for cent in range(k):
            ptsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centriods[cent, :] = mean(ptsInCluster, axis=0)
    return centriods, clusterAssment
