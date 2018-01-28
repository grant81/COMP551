import numpy as np
import matplotlib.pyplot as plt
import math
import csv

m = 20


def getW(transetPath):  # 'Datasets\\Dataset_1_train.csv'
    with open(transetPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVarTemp = []
        targetVarTemp = []
        inputVar = []
        targetVar = []
        one = []
        for row in readCSV:
            inputVarTemp.append(float(row[0]))
            targetVarTemp.append(float(row[1]))
        for i in range(len(inputVarTemp)):
            one.append(1.0)
    for j in range(m, 0, -1):
        currentInput = np.power(inputVarTemp, j)
        inputVar.append(currentInput)

    inputVar.append(one)
    targetVar.append(targetVarTemp)
    inputVar = np.transpose(np.array(inputVar, dtype='float'))
    targetVar = np.transpose(np.array(targetVar, dtype='float'))
    step1 = np.dot(np.transpose(inputVar), inputVar)  # xtx
    step2 = np.dot(np.transpose(inputVar), targetVar)  # xty
    w = np.dot(np.linalg.inv(step1), step2)
    return w


# print(w)

def calTarget(x, w):
    y = 0.0
    for i in range(m, -1, -1):
        y = y + math.pow(x, i) * w[m - i][0]
    return y


def produceResultPoint(testDataPath, w):
    with open(testDataPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVar = []
        targetVar = []
        result = []

        for row in readCSV:
            inputVar.append(float(row[0]))
            targetVar.append(float(row[1]))

        for i in range(len(inputVar)):
            result.append(calTarget(inputVar[i], w))
            if (abs(result[i]) > 60):
                print(inputVar[i], result[i])

    plt.plot(sorted(inputVar), sorted(result), 'b')
    plt.plot(inputVar, targetVar, 'ro')
    plt.title('Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


def calMSE(testDataPath, w):
    with open(testDataPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVarTemp = []
        targetVarTemp = []
        inputVar = []
        targetVar = []
        one = []
        for row in readCSV:
            inputVarTemp.append(float(row[0]))
            targetVarTemp.append(float(row[1]))
        for i in range(len(inputVarTemp)):
            one.append(1.0)
    for j in range(m, 0, -1):
        currentInput = np.power(inputVarTemp, j)
        inputVar.append(currentInput)

    inputVar.append(one)
    targetVar.append(targetVarTemp)
    inputVar = np.transpose(np.array(inputVar, dtype='float'))
    targetVar = np.transpose(np.array(targetVar, dtype='float'))

    YsXw = np.subtract(targetVar, np.dot(inputVar, w))
    # mean square error calculation !!!!!!!!!!!!!!!!!question
    Error = np.square(YsXw).mean()
    return Error


#

# w = getW('Datasets\\Dataset_1_train.csv')
# print(calMSE('Datasets\\Dataset_1_valid.csv'),w)

def getWstar(trainSetPath, lamda):
    with open(trainSetPath) as trainData1:
        readCSV = csv.reader(trainData1, delimiter=',')
        inputVarTemp = []
        targetVarTemp = []
        inputVar = []
        targetVar = []
        one = []
        for row in readCSV:
            inputVarTemp.append(float(row[0]))
            targetVarTemp.append(float(row[1]))
        for i in range(len(inputVarTemp)):
            one.append(1.0)
    for j in range(m, 0, -1):
        currentInput = np.power(inputVarTemp, j)
        inputVar.append(currentInput)

    inputVar.append(one)
    targetVar.append(targetVarTemp)
    inputVar = np.transpose(np.array(inputVar, dtype='float'))
    targetVar = np.transpose(np.array(targetVar, dtype='float'))
    step1 = np.dot(np.transpose(inputVar), inputVar)
    step2 = np.add(step1, lamda * np.identity(step1.shape[0]))
    step3 = np.linalg.inv(step2)
    step4 = np.dot(np.transpose(inputVar), targetVar)
    w = np.dot(step3, step4)
    return w


def findLamda():
    lamda = 0.0
    MSESet = []
    MSESet2 = []
    wSet = []
    lamdas = []
    while (lamda <= 1):
        w = getWstar('Datasets\\Dataset_1_train.csv', lamda)
        wSet.append(w)
        # temp = []
        lamdas.append(lamda)
        MSESet.append(calMSE('Datasets\\Dataset_1_valid.csv', w))
        # MSESet.append(temp)
        lamda += 0.005
    lamda = 0.0
    while (lamda <= 1):
        w = getWstar('Datasets\\Dataset_1_train.csv', lamda)
        # temp = []
        # temp.append(lamda)
        MSESet2.append(calMSE('Datasets\\Dataset_1_train.csv', w))
        # MSESet2.append(temp)
        lamda += 0.005
    plt.plot(lamdas,MSESet, 'r-')
    plt.plot(lamdas,MSESet2, 'b-')
    plt.axis([0, 200, 0, 10])
    plt.show()
    minErr = 0.0
    minLamda = 0.0
    minW = []
    for i in range(len(MSESet)):
        if (minErr == 0.0):
            minErr = MSESet[i]
            minLamda = lamdas[i]
            minW = wSet[i]
        elif (minErr > MSESet[i]):
            minErr = MSESet[i]
            minLamda = lamdas[i]
            minW = wSet[i]
    print(minErr, minLamda)


w = getWstar('Datasets\\Dataset_1_train.csv', 0.02)
w = getW('Datasets\\Dataset_1_train.csv')

produceResultPoint('Datasets\\Dataset_1_valid.csv', w)
findLamda()
# print(calTarget(0.996344491497,w))