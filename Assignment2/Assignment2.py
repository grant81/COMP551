import numpy as np
import matplotlib.pyplot as plt
import math
from random import shuffle
import csv
##Q1
def generateData(covPath,meanPath1,meanPath2):
    with open(covPath) as file:
        readCSV = csv.reader(file, delimiter=',')
        covData = []
        for row in readCSV:
            inputTemp = []
            for i in range(len(row)):
                if(row[i]!= ''):
                    inputTemp.append(float(row[i]))
            covData.append(inputTemp)
    covData = np.array(covData,dtype='float')

    with open(meanPath1) as file:
        readCSV = csv.reader(file, delimiter=',')
        meanData1 = []
        for row in readCSV:
            for i in range(len(row)):
                if (row[i] != ''):
                    meanData1.append(float(row[i]))
    meanData1 = np.array(meanData1,dtype='float')

    with open(meanPath2) as file:
        readCSV = csv.reader(file, delimiter=',')
        meanData2 = []
        for row in readCSV:
            for i in range(len(row)):
                if (row[i] != ''):
                    meanData2.append(float(row[i]))
    meanData2 = np.array(meanData2,dtype='float')

    class1 = np.random.multivariate_normal(meanData1,covData,2000)
    c1 = -1* np.ones((2000,21),dtype='float')
    c1[:,:-1] = class1
    class2 = np.random.multivariate_normal(meanData2,covData,2000)
    c2 = np.ones((2000,21),dtype='float')
    c2[:,:-1]= class2
    np.random.shuffle(c1)
    np.random.shuffle(c2)
    testSet = []
    trainSet = []
    for i in range(2000):
        if (i<600):
            testSet.append(c1[i])
            testSet.append(c2[i])
        else:
            trainSet.append(c1[i])
            trainSet.append(c2[i])
    shuffle(testSet)
    shuffle(trainSet)
    with open('hwk2_datasets_corrected\\DS1_test.csv', 'w', newline='') as file:
        csvWriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for j in range(len(testSet)):
            csvWriter.writerow(testSet[j])
    with open('hwk2_datasets_corrected\\DS1_train.csv', 'w', newline='') as file:
        csvWriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for j in range(len(trainSet)):
            csvWriter.writerow(trainSet[j])

# generateData('hwk2_datasets_corrected\\DS1_Cov.txt','hwk2_datasets_corrected\\DS1_m_0.txt','hwk2_datasets_corrected\\DS1_m_1.txt')