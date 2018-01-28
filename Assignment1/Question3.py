import numpy as np
import matplotlib.pyplot as plt

import csv
from random import shuffle

def prepareData(dataPath, statPath):
    with open(dataPath) as dataPath:
        readCSV = csv.reader(dataPath, delimiter=',')
        data = []
        for row in readCSV:
            data.append(row)
    with open(statPath) as statPath:
        readCSV = csv.reader(statPath, delimiter=',')
        stat = []
        for row in readCSV:
            stat.append(row)
        print(len(data) / 5, len(data[0]))

    for j in range(len(data)):
        for i in range(5, len(data[0])):
            if (data[j][i] == '?'):
                data[j][i] = stat[i - 5][6]
    shuffle(data)
    dataPortions = []

    c = 0
    for i in range(0, len(data), 399):
        c += 1
        if (c < 5):
            dataPortions.append(data[i:i + 399])
        else:
            dataPortions.append(data[i:len(data)])
    for i in range(1, 6):
        with open('Datasets\\CandC-train' + str(i) + '.csv', 'w', newline='') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for j in range(len(dataPortions)):
                if (i - 1 != j):
                    for k in range(len(dataPortions[j])):
                        csvWriter.writerow(dataPortions[j][k])
        with open('Datasets\\CandC-test' + str(i) + '.csv', 'w', newline='') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k in range(len(dataPortions[i - 1])):
                csvWriter.writerow(dataPortions[i - 1][k])


def getW(trainPath):
    with open(trainPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []

        for row in readCSV:
            inputVarTemp = []
            for i in range(5, len(row) - 1):
                inputVarTemp.append(float(row[i]))
            inputVarTemp.append(1.0)
            inputVar.append(inputVarTemp)
            targetVarTemp = []
            targetVarTemp.append(float(row[len(row) - 1]))
            targetVar.append(targetVarTemp)

    inputVar = np.array(inputVar, dtype='float')
    targetVar = (np.array(targetVar, dtype='float'))
    step1 = np.dot(np.transpose(inputVar), inputVar)  # xtx
    step2 = np.dot(np.transpose(inputVar), targetVar)  # xty
    w = np.dot(np.linalg.inv(step1), step2)
    return w


def getWRidge(trainSetPath, lamda):
    with open(trainSetPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []

        for row in readCSV:
            inputVarTemp = []
            for i in range(5, len(row) - 1):
                inputVarTemp.append(float(row[i]))
            inputVarTemp.append(1.0)
            inputVar.append(inputVarTemp)
            targetVarTemp = []
            targetVarTemp.append(float(row[len(row) - 1]))
            targetVar.append(targetVarTemp)

    inputVar = np.array(inputVar, dtype='float')
    targetVar = (np.array(targetVar, dtype='float'))
    step1 = np.dot(np.transpose(inputVar), inputVar)
    step2 = np.add(step1, lamda * np.identity(step1.shape[0]))
    step3 = np.linalg.inv(step2)
    step4 = np.dot(np.transpose(inputVar), targetVar)
    w = np.dot(step3, step4)
    return w


def produceResultPoint(testDataPath, w):
    with open(testDataPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []

        for row in readCSV:
            inputVarTemp = []
            for i in range(5, len(row) - 1):
                inputVarTemp.append(float(row[i]))
            inputVarTemp.append(1.0)
            inputVar.append(inputVarTemp)
            targetVarTemp = []
            targetVarTemp.append(float(row[len(row) - 1]))
            targetVar.append(targetVarTemp)
    inputVar = np.array(inputVar, dtype='float')
    targetVar = (np.array(targetVar, dtype='float'))
    result = np.dot(inputVar, w)
    # for i in range(len(targetVar)):
    #     print(targetVar[i],result[i])
    plt.plot(result, 'b^')
    plt.plot(targetVar, 'ro')
    plt.show()


def calMSE(testDataPath, w):
    with open(testDataPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []

        for row in readCSV:
            inputVarTemp = []
            for i in range(5, len(row) - 1):
                inputVarTemp.append(float(row[i]))
            inputVarTemp.append(1.0)
            inputVar.append(inputVarTemp)
            targetVarTemp = []
            targetVarTemp.append(float(row[len(row) - 1]))
            targetVar.append(targetVarTemp)
    inputVar = np.array(inputVar, dtype='float')
    targetVar = (np.array(targetVar, dtype='float'))

    YsXw = np.subtract(targetVar, np.dot(inputVar, w))
    Error = np.square(YsXw).mean()
    return Error


def findLamda():
    lamda = 0.0
    MSEAverage = []
    lamdas = []
    while (lamda <= 5):
        MSEs = []
        for i in range(1, 6):
            w = getWRidge('Datasets\\CandC-train' + str(i) + '.csv', lamda)
            mse = calMSE('Datasets\\CandC-test' + str(i) + '.csv', w)
            MSEs.append(mse)
        meanMSE = np.array(MSEs, dtype='float').mean()
        print(meanMSE)
        MSEAverage.append(meanMSE)
        lamdas.append(lamda)
        lamda += 0.025
    minErr = 0.0
    minLamda = 0.0
    for i in range(len(lamdas)):
        if (minErr == 0.0):
            minErr = MSEAverage[i]
            minLamda = lamdas[i]
        elif (minErr > MSEAverage[i]):
            minErr = MSEAverage[i]
            minLamda = lamdas[i]

    print(minErr, minLamda)
    plt.axis([0, 5, 0, 10])
    plt.plot(lamdas, MSEAverage)
    plt.show()


# MSEs = []
# Ws = []
#
# for i in range(1, 6):
#     w = getWRidge('Datasets\\CandC-train' + str(i) + '.csv', 5)
#     # Ws.append(w)
#     produceResultPoint('Datasets\\CandC-test' + str(i) + '.csv', w)
#     mse = calMSE('Datasets\\CandC-test' + str(i) + '.csv', w)
#     print(mse)
#     MSEs.append(mse)
#     # findLamda()
Ws = []

for i in range(1, 6):
    w = getWRidge('Datasets\\CandC-train' + str(i) + '.csv', 5)
    Ws.append(np.array(w))
for j in range(0,123):
    print(Ws[0][j][0],Ws[1][j][0],Ws[2][j][0],Ws[3][j][0],Ws[4][j][0])
    print(np.array(Ws).mean())