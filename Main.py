import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize as tok
from scipy import linalg as SPLA
import utils

testFile = r"C:\Users\Carl Wilhjelm\PycharmProjects\ICSProject\logs\syslogTest.txt"

# create tokenized list of all logs
syslog = []
with open(testFile, 'r') as f:
    syslogText = f.readlines()

for line in syslogText:
    syslog.append(tok(line))
n = len(syslog)
print(n)
opNumber = int(n * (0.01))

# create naive distance matrix for all logs
distanceMatrix = [[0 for x in range(n)] for y in range(n)]
for i in range(n):
    for j in range(i, n):
        distance = 0
        l = len(syslog[i])
        m = len(syslog[j])
        if m > l:
            k = l
        else:
            k = m

        for e in range(k):
            if syslog[i][e] != syslog[j][e]:
                distance += 1
        distanceMatrix[i][j] = distance / k
        distanceMatrix[j][i] = distanceMatrix[i][j]

cov = np.cov(distanceMatrix, rowvar=False)

k = 2
w, v = SPLA.eigh(cov, eigvals=(n - k, n - 1))
U = np.array([v.T[1], v.T[0]])
distanceMatrix = np.array(distanceMatrix)
TransformedData = np.dot(U, distanceMatrix.T)


dataSet = TransformedData.T
# 3 is the k for k-means. dataset is the nx2 matrix after PCA
centroids, clusterAssment = utils.kMeans(dataSet, 3, syslog)
print('cluster center are:')
print(centroids)
# clusterAssment is a list like [[1,12],[2,13],[3,10]],
# in which 1,2,3 are cluster and 12,13,10 is the distance from cluster center to the data point
a = np.array(clusterAssment)
# sortedList is a list contain index. like [13,14, 1 ,2.....]
# 13 means the first element in a is the 13rd largest one.
sortedList = list(np.argsort(a[:, 1]),)
topList = []

# opNumber is the 5% of data points
for i in range(opNumber):
    # index(n-1-i), I want to get the 999th,998th,997th's index in the sortedList
    index = sortedList.index(n - 1 - i)
    # then I use the index to get the  999th,998th,997th largest distance in the dataset.
    topList.append(dataSet[index])
    print("top record is:")
    # syslog[i] is the ith record in original data
    print(dataSet[index], syslog[i])

outliers = np.array(topList)
print("\noutliers")
print(outliers)

# plot each point (x(n), y(n))
print()
print(TransformedData)
print(TransformedData.shape)
print(TransformedData.T[0, :])
plt.plot(TransformedData[0, :], TransformedData[1, :], 'go')
plt.plot(outliers[:, 0], outliers[:, 1], 'ro')
plt.plot(centroids[:, 0], centroids[:, 1], 'bo')
plt.show()
