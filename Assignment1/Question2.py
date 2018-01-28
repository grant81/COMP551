import numpy as np
import matplotlib.pyplot as plt
import math
import csv
stepSize = 1e-6
def getWByGD(trainPath,stepSize,numStep):#'Datasets\\Dataset_2_train.csv'
    with open(trainPath) as trainData:
        readCSV = csv.reader(trainData, delimiter=',')
        xTemp = []
        x = []
        y = []
        one = []

        for row in readCSV:
            xTemp.append(float(row[0]))
            y.append(float(row[1]))
        for i in range(len(xTemp)):
            one.append(1.0)


        x.append(xTemp)
        x.append(one)
        y = np.transpose(np.array(y, dtype='float'))
        x = np.transpose(np.array(x,dtype='float'))

    w= np.array([1.0,1.0],dtype='float')
    MSEListValid = []
    MSEListTrain = []
    with open('Datasets\\Dataset_2_valid.csv') as validData:
        readCSV = csv.reader(validData, delimiter=',')
        xTemp = []
        one =[]
        x1 = []
        y1 = []

        for row in readCSV:
            xTemp.append(float(row[0]))
            y1.append(float(row[1]))

        for i in range(len(xTemp)):
            one.append(1.0)

        x1.append(xTemp)
        x1.append(one)
        y1 = np.transpose(np.array(y1, dtype='float'))
        x1 = np.transpose(np.array(x1,dtype='float'))

    for i in range (numStep):
        for j in range (x.shape[0]):
            w[1] = w[1] - stepSize*(w[1]+w[0]*x[j][0] - y[j])
            w[0] = w[0] - stepSize * (w[1] + w[0] * x[j][0] - y[j])*x[j][0]

        MSEListValid.append(calMSE(x1,y1,w))
        MSEListTrain.append(calMSE(x,y,w))
    print(MSEListTrain)
    print(MSEListValid)
    plt.plot(MSEListValid,'r')
    plt.plot(MSEListTrain,'b')
    plt.axis([0,20000,0,30])
    plt.show()

    return w
def getWByGDNoGraph(trainPath,stepSize,numStep):#'Datasets\\Dataset_2_train.csv'
    with open(trainPath) as trainData:
        readCSV = csv.reader(trainData, delimiter=',')
        xTemp = []
        x = []
        y = []
        one = []

        for row in readCSV:
            xTemp.append(float(row[0]))
            y.append(float(row[1]))
        for i in range(len(xTemp)):
            one.append(1.0)


        x.append(xTemp)
        x.append(one)
        y = np.transpose(np.array(y, dtype='float'))
        x = np.transpose(np.array(x,dtype='float'))

    w= np.array([1.0,1.0],dtype='float')
    for i in range (numStep):
        for j in range (x.shape[0]):
            w[1] = w[1] - stepSize*(w[1]+w[0]*x[j][0] - y[j])
            w[0] = w[0] - stepSize * (w[1] + w[0] * x[j][0] - y[j])*x[j][0]

    return w
def calTarget (x , w):
    y = w[1] + x * w[0]

    return y

def produceResultPoint (testDataPath,w):
    with open(testDataPath) as trainData1:
        readCSV = csv.reader(trainData1,delimiter=',')
        inputVar = []
        targetVar = []
        result = []

        for row in readCSV:
            inputVar.append(float(row[0]))
            targetVar.append(float(row[1]))
        for i in range(len(inputVar)):
            result.append(calTarget(inputVar[i],w))

    plt.plot(inputVar,result,'g-')
    plt.plot(inputVar,targetVar,'ro')
    plt.show()


def calMSE (x,y,w):
    YsXw = np.subtract(np.dot(x,w),y)
    Error = np.square(YsXw).mean()
    return Error
w = getWByGD('Datasets\\Dataset_2_train.csv',stepSize,20000)
for i in range(1,12):
    stepSize = 0.000125
    stepSize = stepSize*i
    print( stepSize)
    w = getWByGD('Datasets\\Dataset_2_train.csv',stepSize,8000)


print("MSE = "+str(calMSE('Datasets\\Dataset_2_valid.csv',w)))

for i in range (1,15,3):
    stepSize = 1e-6
    stepNum = i * 1000
    w = getWByGDNoGraph('Datasets\\Dataset_2_train.csv',stepSize,stepNum)
produceResultPoint('Datasets\\Dataset_2_valid.csv',w)
