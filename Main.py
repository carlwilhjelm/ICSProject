import re
from nltk import word_tokenize as tok

localDir = "D:\School\GSU\Fall 2018\Intro to Cybersecurity\project\\"
testFile = localDir + "syslogTest.txt"

reDateTime = re.compile(r"^.*?(?= alina)")

# create tokenized list of all logs
syslog = []
with open(testFile, 'r') as f:
    for line in f:
        syslog.append(tok(f.readline()))

n = len(syslog)

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
        distanceMatrix[i][j] = distance/k
        distanceMatrix[j][i] = distanceMatrix[i][j]