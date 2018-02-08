import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
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
    c1 = -1* np.ones((2000,21),dtype='float') #-1 when using u0
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



def LDA(trainingPath):
    with open(trainingPath) as file:
        readCSV = csv.reader(file, delimiter=',')
        X = []
        for row in readCSV:
            inputVarTemp = []
            for i in range(len(row)):
                inputVarTemp.append(float(row[i]))
            X.append(inputVarTemp)
    X = np.array(X,dtype='float')
    sum1 = np.zeros(X[0].shape,dtype='float')
    sum2 = np.zeros(X[0].shape,dtype='float')
    N0 = 0
    N1 = 0
    for i in range(len(X)):
        if (X[i][len(X[0])-1]==-1):
            sum1 = np.add(sum1,X[i])
            N0 +=1
        elif (X[i][len(X[0])-1]==1):
            sum2 = np.add(sum2,X[i])
            N1 +=1
    mean1 = np.divide(sum1,N0)
    mean2 = np.divide(sum2,N1)
    mean1 = mean1[:-1].reshape(1,-1)
    mean2 = mean2[:-1].reshape(1,-1)

    sum = np.zeros((len(mean1[0]),len(mean1[0])))
    for i in range(len(X)):
        step1 = np.subtract(X[i][:-1],mean1)
        step2 = np.subtract(X[i][:-1],mean2)
        step3 = np.dot(np.transpose(step1),step1)
        step4 = np.dot(np.transpose(step2),step2)
        sum = np.add(np.add(step3,step4),sum)

    cov = np.divide(sum,N0+N1)
    covInverse = np.linalg.inv(cov)
    w1 = np.dot(covInverse,np.transpose(np.subtract(mean1,mean2)))
    term1 = 1/2*np.dot(np.dot(mean1,covInverse),np.transpose(mean1))
    term2 = 1/2*np.dot(np.dot(mean2,covInverse),np.transpose(mean2))
    w0 = np.subtract(term2,term1)
    w = [w0,w1]
    with open('hwk2_datasets_corrected\\ParameterLearntDS1.txt', 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=' ')
        csvWriter.writerow('w0:')
        csvWriter.writerow(w0)
        csvWriter.writerow('w1:')
        csvWriter.writerow(w1)
    return w


def testLDA (testFilePath,w):
    with open(testFilePath) as file:
        readCSV = csv.reader(file, delimiter=',')
        X = []
        for row in readCSV:
            inputVarTemp = []
            for i in range(len(row)):
                inputVarTemp.append(float(row[i]))
            X.append(inputVarTemp)
    X = np.array(X,dtype='float')
    result = []
    hit = 0.0
    c1 = 0
    c2 = 0
    c1c =0
    c2c =0
    w0 = w[0]
    w1 = w[1]
    for i in range(len(X)):
        ans = w0 + np.dot(X[i][:-1],w1)
        if (ans>0):
            result.append(-1)
            if (result[i] == X[i][len(X[0]) - 1]):
                hit += 1
                c1c +=1
            c1 +=1
        else:
            result.append(1)
            if (result[i] == X[i][len(X[0]) - 1]):
                hit += 1
                c2c +=1
            c2 +=1

    print(hit/len(X))
    print(c1, c1c)
    print(c2, c2c)
# generateData('hwk2_datasets_corrected\\DS1_Cov.txt','hwk2_datasets_corrected\\DS1_m_0.txt','hwk2_datasets_corrected\\DS1_m_1.txt')
# w = LDA('hwk2_datasets_corrected\\DS1_train.csv')
# testLDA('hwk2_datasets_corrected\\DS1_test.csv',w)

def kNN (trainX,inputX,k):

    distanceSet = []
    for i in range (len(trainX)):
        sum = 0
        for j in range(len(inputX)-1):
            sum += (trainX[i][j]-inputX[j])**2
        distanceSet.append([np.sqrt(sum),i,trainX[i][len(trainX[0])-1]])
    distanceSet = sorted(distanceSet)
    neighbors = []
    for i in range (k):
        neighbors.append(distanceSet[i][2])
    result = np.sum(neighbors)
    if (result>0):
        return 1
    else:
        return -1

def testKNN (trainingSet, testingSet):
    with open(trainingSet) as file:
        readCSV = csv.reader(file, delimiter=',')
        X = []
        for row in readCSV:
            inputVarTemp = []
            for i in range(len(row)):
                inputVarTemp.append(float(row[i]))
            X.append(inputVarTemp)
    X = np.array(X,dtype='float')
    X = preprocessing.normalize(X,axis=0)
    X[:, :-1] = preprocessing.normalize(X[:, :-1], axis=0)
    with open(testingSet) as file:
        readCSV = csv.reader(file, delimiter=',')
        XTest = []
        for row in readCSV:
            inputVarTemp = []
            for i in range(len(row)):
                inputVarTemp.append(float(row[i]))
            XTest.append(inputVarTemp)
    XTest = np.array(XTest,dtype='float')


    for j in range(3,17,2):
        hit = 0
        c1Count = 0
        c2Count = 0
        for i in range(len(XTest)):
            ans = kNN(X,XTest[i],j)
            if (ans ==  XTest[i][len(XTest[0])-1]):
                hit +=1
            if (ans >0):
                c2Count +=1
            else:
                c1Count +=1
        print('k = '+str(j))
        print(hit/len(XTest))
        print(c1Count,c2Count)

testKNN('hwk2_datasets_corrected\\DS1_train.csv','hwk2_datasets_corrected\\DS1_test.csv')
