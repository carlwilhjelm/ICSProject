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
print(distanceMatrix)
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
centroids, clusterAssment = utils.kMeans(dataSet, 3)
print(centroids)
