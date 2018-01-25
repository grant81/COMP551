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

        x.append(one)
        x.append(xTemp)
        y = np.transpose(np.array(y, dtype='float'))
        x = np.transpose(np.array(x,dtype='float'))

    step1 = np.dot(np.transpose(x),x)
    step2 = np.dot(np.transpose(x),y)
    w = np.dot(np.linalg.inv(step1),step2)
    MSEList = []
    with open('Datasets\\Dataset_2_train.csv') as validData:
        readCSV = csv.reader(validData, delimiter=',')
        xTemp = []
        x1 = []
        y1 = []

        for row in readCSV:
            xTemp.append(float(row[0]))
            y1.append(float(row[1]))


        x1.append(one)
        x1.append(xTemp) #(1,x)
        y1 = np.transpose(np.array(y1, dtype='float'))
        x1 = np.transpose(np.array(x1,dtype='float'))

    for i in range (numStep):
        for j in range (x.shape[0]):
            w[0] = w[0] - stepSize*(w[0]+w[1]*x[j][1] - y[j])
            w[1] = w[1] - stepSize * (w[0] + w[1] * x[j][1] - y[j])*x[j][1]
        temp = []
        temp.append(i)
        temp.append(calMSE(x1,y1,w))
        MSEList.append(temp)
    print(MSEList)
    plt.plot(MSEList,'ro')
    plt.axis([0,20000,0.09,0.2])
    plt.show()

    return w

def calTarget (x , w):
    y = w[0] + x * w[1]

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

    plt.plot(inputVar,result,'g^')
    plt.plot(inputVar,targetVar,'ro')
    plt.show()


def calMSE (x,y,w):
    # with open(testDataPath) as trainData:
    #     readCSV = csv.reader(trainData, delimiter=',')
    #     xTemp = []
    #     x = []
    #     y = []
    #     one = []
    #     for row in readCSV:
    #         xTemp.append(float(row[0]))
    #         y.append(float(row[1]))
    #     for i in range(len(xTemp)):
    #         one.append(1.0)
    #     x.append(xTemp)
    #     x.append(one)
    #     y = np.transpose(np.array(y, dtype='float'))
    #     x = np.transpose(np.array(x,dtype='float'))
    YsXw = np.subtract(np.dot(x,w),y)

    Error = np.square(YsXw).mean()
    return Error
w = getWByGD('Datasets\\Dataset_2_train.csv',stepSize,20000)
# print("MSE = "+str(calMSE('Datasets\\Dataset_2_valid.csv',w)))
produceResultPoint('Datasets\\Dataset_2_valid.csv',w)
