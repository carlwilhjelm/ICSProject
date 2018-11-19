import re
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize as tok
from numpy import *
from scipy import linalg as SPLA
import utils

# localDir = "D:\School\GSU\Fall 2018\Intro to Cybersecurity\project\\"
testFile = "syslogTest.txt"

reDateTime = re.compile(r"^.*?(?= alina)")

# create tokenized list of all logs
syslog = []
with open(testFile, 'r') as f:
    syslogText = f.readlines()

for line in syslogText:
    syslog.append(tok(line))
    # print(line)
n = len(syslog)
print(n)
# print(syslog)
# n = 100
opNumber = int(n * (0.05))
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
            if syslog[i][e] is not syslog[j][e]:
                distance += 1
        distanceMatrix[i][j] = distance / k
        distanceMatrix[j][i] = distanceMatrix[i][j]

avg = np.sum(distanceMatrix, axis=0)
for i in range(n):
    avg[i] /= n
for i in range(n):
    for j in range(n):
        distanceMatrix[i][j] -= avg[j]
cov = np.cov(distanceMatrix, rowvar=False)
k = 2
w, v = SPLA.eigh(cov, eigvals=(n - k, n - 1))
U = np.array([v.T[1], v.T[0]])
distanceMatrix = array(distanceMatrix)
TransformedData = np.dot(U, distanceMatrix.T)
# plot each point (x(n), y(n))
# print(TransformedData)
# print(TransformedData.shape)
# print(TransformedData.T[0, :])
# plt.plot(TransformedData[0, :], TransformedData[1, :], 'ro')
# plt.show()
dataSet = TransformedData.T
print(dataSet)
centroids, clusterAssment = utils.kMeans(dataSet, 3,syslog)# 3 is the k for k-means. dataset is the nx2 matrix after PCA
print('cluster center are:')
print(centroids)
a = array(clusterAssment) # clusterAssment is a list like [[1,12],[2,13],[3,10]], in which 1,2,3 are cluster and 12,13,10 is the distance from cluster center to the data point
sortedList = list(argsort(a[:, 1]),)# sortedList is a list contain index. like [13,14, 1 ,2.....] 13 means the first element in a is the 13rd largest one.
topList = []

for i in range(opNumber):# opNumber is the 5% of data points
    index = sortedList.index(n - 1 - i)# index(n-1-i), I want to get the 999th,998th,997th's index in the sortedList
    topList.append(dataSet[index])# then I use the index to get the  999th,998th,997th largest distance in the dataset.
    print("top record is:")
    print(dataSet[index],syslog[i])# syslog[i] is the ith record in original data
# print(sortedList)
# index = np.where(sortedList == 97)
# index.getfield(array)
# print(index)
# for i in range(opNumber)；
#     sortedList.__index__()
# for i in range(shape(dataSet)[0]):
