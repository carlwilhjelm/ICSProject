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
def kMeans(dataSet, k, syslog, distMeas=distEclud, createCenter=randCenter):
    m = shape(dataSet)[0]  # row of dataset,dataset is [x,y]
    clusterAssment = mat(zeros((m, 3)))  # create matrix [cluster,distance, index of syslog] like [0,19, 2]
    centriods = createCenter(dataSet, k)
    clusterChange = True
    while clusterChange:
        clusterChange = False
        for i in range(m):
            minDist = inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centriods[j, 0:1], dataSet[i, 0:1])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True;
            clusterAssment[i, :] = minIndex, minDist ** 2, i
        print(centriods)
        for cent in range(k):
            ptsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centriods[cent, :] = mean(ptsInCluster, axis=0)
    # the clusterAssment is a nx3 matrix, each record is like [0,19,2] which means
    # this record is in cluster 0, distance from cluster center=19, it is the 2rd row record in original syslog dataset
    return centriods, clusterAssment
