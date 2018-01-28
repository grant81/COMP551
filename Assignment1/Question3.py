import numpy as np
import matplotlib.pyplot as plt
import math
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
    plt.plot(result, 'b^',label = 'result')
    plt.plot(targetVar, 'ro', label = 'target')
    plt.legend()
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
    lamda = 1e-9
    MSEAverage = []
    lamdas = []
    Ws = []
    while (lamda <= 10000):
        MSEs = []
        Wtemp = []
        for i in range(1, 6):
            w = getWRidge('Datasets\\CandC-train' + str(i) + '.csv', lamda)
            mse = calMSE('Datasets\\CandC-test' + str(i) + '.csv', w)
            MSEs.append(mse)
            Wtemp.append(w)
        meanMSE = np.array(MSEs, dtype='float').mean()
#         print(meanMSE)
        MSEAverage.append(meanMSE)
        lamdas.append(math.log10(lamda))
        Ws.append(Wtemp)
        lamda = lamda * 10
    minErr = 0.0
    minLamda = 0.0
    for i in range(len(lamdas)):
        if (minErr == 0.0):
            minErr = MSEAverage[i]
            minLamda = lamdas[i]
        elif (minErr > MSEAverage[i]):
            minErr = MSEAverage[i]
            minLamda = 10**lamdas[i]
            minW = Ws[i]
    print('the optimal lamda is = '+str(minLamda)+', produce an Average MSE of '
             +str(minErr))
    # plt.axis([0, 13, 0, 8])

    plt.plot(lamdas, MSEAverage)
    plt.title('MSE vs Lamda')
    plt.xlabel('log(Lamda)')
    plt.ylabel('MSE')
    plt.show()


# MSEs = []

#
# for i in range(1, 6):
#     w = getWRidge('Datasets\\CandC-train' + str(i) + '.csv', 5)
#     # Ws.append(w)
#     produceResultPoint('Datasets\\CandC-test' + str(i) + '.csv', w)
#
# # findLamda()


def getWRidgeFS(trainSetPath,lamda,removedFeatures):
    with open(trainSetPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []

        for row in readCSV:
            inputVarTemp = []
            for i in range(5, len(row) - 1):
                if ((i == removedFeatures).any()):
                    pass
                else:
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

def produceResultPointFS(testDataPath, w,removedFeatures):
    with open(testDataPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []

        for row in readCSV:
            inputVarTemp = []
            for i in range(5, len(row) - 1):
                if ((i == removedFeatures).any()):
                    pass
                else:
                    inputVarTemp.append(float(row[i]))
            inputVarTemp.append(1.0)
            inputVar.append(inputVarTemp)
            targetVarTemp = []
            targetVarTemp.append(float(row[len(row) - 1]))
            targetVar.append(targetVarTemp)
    inputVar = np.array(inputVar, dtype='float')
    targetVar = (np.array(targetVar, dtype='float'))
    result = np.dot(inputVar, w)
    plt.plot(result, 'b^',label = 'result')
    plt.plot(targetVar, 'ro', label = 'target')
    plt.legend()
    plt.show()


def calMSEFS(testDataPath, w ,removedFeatures):
    with open(testDataPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []

        for row in readCSV:
            inputVarTemp = []
            for i in range(5, len(row) - 1):
                if ((i == removedFeatures).any()):
                    pass
                else:
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

Ws = []
for i in range(1, 6):
    w = getWRidge('Datasets\\CandC-train' + str(i) + '.csv', 1.0)
    Ws.append(np.array(w))

WAvg = []
for j in range(0, 123):
    rowTemp = []
    sum = 0.0
    rowTemp.append(Ws[0][j][0])
    rowTemp.append(Ws[1][j][0])
    rowTemp.append(Ws[2][j][0])
    rowTemp.append(Ws[3][j][0])
    rowTemp.append(Ws[4][j][0])

    for k in range(len(rowTemp)):
        sum += abs(rowTemp[k])
    WAvg.append(sum/len(rowTemp))
WAvg = np.array(WAvg,dtype='float')
smallValues = WAvg.argsort()[0:22]

# smallValues = np.array(smallValues.tolist())
# print(smallValues)


for i in range(1, 6):
    w = getWRidgeFS('Datasets\\CandC-train' + str(i) + '.csv', 1.0,smallValues)
    mse = calMSEFS('Datasets\\CandC-test' + str(i) + '.csv', w,smallValues)
    print('the MSE for 80-20 split ' + str(i) + ' with feature selection is ' + str(mse))
    produceResultPointFS('Datasets\\CandC-test' + str(i) + '.csv', w,smallValues)
