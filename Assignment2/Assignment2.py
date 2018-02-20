import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from random import shuffle
import csv
##Q1
def CSVfileReader(filePath):
    with open(filePath) as file:
        readCSV = csv.reader(file, delimiter=',')
        data = []
        for row in readCSV:
            inputTemp = []
            for i in range(len(row)):
                if(row[i]!= ''):
                    inputTemp.append(float(row[i]))
            data.append(inputTemp)
    data = np.array(data,dtype='float')
    return data

def generateData1(covPath,meanPath1,meanPath2):
    covData = CSVfileReader(covPath)
    meanData1 = CSVfileReader(meanPath1)
    meanData2 = CSVfileReader(meanPath2)
    class1 = np.random.multivariate_normal(meanData1[0],covData,2000)
    c1 = -1* np.ones((2000,21),dtype='float') #-1 when using u0
    c1[:,:-1] = class1
    class2 = np.random.multivariate_normal(meanData2[0],covData,2000)
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
    X = CSVfileReader(trainingPath)
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
    P1 = N0/len(X)
    P2 = N1/len(X)
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
    w0 = np.subtract(term2,term1) + np.log(P1)- np.log(P2)
    w = [w0,w1]
    print('parameter learnt')
    print('w0 = '+str(w0[0][0]))
    print('w1 = '+str(w1.T[0]))
    return w


def testLDA (testFilePath,w):
    X = CSVfileReader(testFilePath)
    result = []
    TPTN = 0.0
    TPFP = 0
    TNFN = 0
    TP =0
    TN =0
    w0 = w[0]
    w1 = w[1]
    for i in range(len(X)):
        ans = w0[0][0] + np.dot(X[i][:-1],w1)
        if (ans>0):
            result.append(-1)
            if (result[i] == X[i][len(X[0]) - 1]):
                TPTN += 1
                TN +=1
            TNFN +=1
        else:
            result.append(1)
            if (result[i] == X[i][len(X[0]) - 1]):
                TPTN += 1
                TP +=1
            TPFP +=1
    accuracy = TPTN/(TPFP+TNFN)
    precision = TP/TPFP
    recall = TP/(TP+TNFN-TN)
    print('treating class 2 as positive')
    print('Accuracy = '+str(accuracy))
    print('Precision = '+str(precision))
    print('Recall = '+str(recall))
    print('F1 Measure = '+ str(2*precision*recall/(precision+recall)))
# generateData('hwk2_datasets_corrected\\DS1_Cov.txt','hwk2_datasets_corrected\\DS1_m_0.txt','hwk2_datasets_corrected\\DS1_m_1.txt')
# w = LDA('hwk2_datasets_corrected\\DS1_train.csv')
# testLDA('hwk2_datasets_corrected\\DS1_test.csv',w)

def kNN (trainX,inputX,k):

    distanceSet = []
    for i in range (len(trainX)):
        sum = 0
        for j in range(len(inputX)-1):
            sum += np.square(trainX[i][j]-inputX[j])
        distanceSet.append([np.sqrt(sum),i,trainX[i][len(trainX[0])-1]])
    distanceSet = sorted(distanceSet)
    neighbors = []
    for i in range (k):
        neighbors.append(distanceSet[i][2])

    result = np.sum(neighbors)
    if (result > 0):
        return 1
    else:
        return -1

def applyKNN (trainingSet, testingSet,k):
    X = CSVfileReader(trainingSet)
    X[:, :-1] = preprocessing.normalize(X[:, :-1], axis=0)
    XTest = CSVfileReader(testingSet)
    XTest[:,:-1] = preprocessing.normalize(XTest[:,:-1],axis=0)
    TPTN = 0.0
    TPFP = 0
    TNFN = 0
    TP = 0
    TN = 0
    #run again the best case
    for i in range(len(XTest)):
        ans = kNN(X, XTest[i], k)
        if (ans > 0):
            if (ans == X[i][len(X[0]) - 1]):
                TPTN += 1
                TN += 1
            TNFN += 1
        else:
            if (ans == X[i][len(X[0]) - 1]):
                TPTN += 1
                TP += 1
            TPFP += 1
    accuracy = TPTN/(TPFP+TNFN)
    precision = TP/TPFP
    recall = TP/(TP+TNFN-TN)
    print('K = '+str(k))
    print('Accuracy = '+str(accuracy))
    print('Precision = '+str(precision))
    print('Recall = '+str(recall))
    print('F1 Measure = '+ str(2*precision*recall/(precision+recall)))



def testKNN (trainingSet, testingSet):
    X = CSVfileReader(trainingSet)
    X[:, :-1] = preprocessing.normalize(X[:, :-1], axis=0)
    XTest = CSVfileReader(testingSet)
    XTest[:,:-1] = preprocessing.normalize(XTest[:,:-1],axis=0)
    minF1 = 1
    minK = 0
    for j in range (1,21,2):
        TPTN = 0.0
        TPFP = 0
        TNFN = 0
        TP = 0
        TN = 0
        for i in range(len(XTest)):
            ans = kNN(X,XTest[i],j)
            if (ans>0):
                if (ans == X[i][len(X[0]) - 1]):
                    TPTN += 1
                    TN +=1
                TNFN +=1
            else:
                if (ans == X[i][len(X[0]) - 1]):
                    TPTN += 1
                    TP +=1
                TPFP +=1
        precision = TP / TPFP
        recall = TP / (TP + TNFN - TN)
        F1 = 2 * precision * recall / (precision + recall)
        print('k = '+str(j))
        print('F1 Measure = ' + str(F1))
        if(F1 < minF1):
            minF1 = F1
            minK = j
    TPTN = 0.0
    TPFP = 0
    TNFN = 0
    TP = 0
    TN = 0
    #run again the best case
    for i in range(len(XTest)):
        ans = kNN(X, XTest[i], minK)
        if (ans > 0):
            if (ans == X[i][len(X[0]) - 1]):
                TPTN += 1
                TN += 1
            TNFN += 1
        else:
            if (ans == X[i][len(X[0]) - 1]):
                TPTN += 1
                TP += 1
            TPFP += 1
    accuracy = TPTN/(TPFP+TNFN)
    precision = TP/TPFP
    recall = TP/(TP+TNFN-TN)
    print('treating class 2 as positive')
    print('the Best K is ' + str((minK)))
    print('Accuracy = '+str(accuracy))
    print('Precision = '+str(precision))
    print('Recall = '+str(recall))
    print('F1 Measure = '+ str(2*precision*recall/(precision+recall)))

def testKNNSci (trainingSet, testingSet):
    with open(trainingSet) as file:
        readCSV = csv.reader(file, delimiter=',')
        X = []
        for row in readCSV:
            inputVarTemp = []
            for i in range(len(row)):
                inputVarTemp.append(float(row[i]))
            X.append(inputVarTemp)
    X = np.array(X,dtype='float')
    X[:, :-1] = preprocessing.normalize(X[:, :-1], axis=0)
    Y = np.transpose(X)[20]

    with open(testingSet) as file:
        readCSV = csv.reader(file, delimiter=',')
        XTest = []
        for row in readCSV:
            inputVarTemp = []
            for i in range(len(row)):
                inputVarTemp.append(float(row[i]))
            XTest.append(inputVarTemp)
    XTest = np.array(XTest,dtype='float')
    XTest[:,:-1] = preprocessing.normalize(XTest[:,:-1],axis=0)
    # neigh = KNeighborsClassifier(n_neighbors=3)
    # neigh.fit(X[:,:-1], Y)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X[:,:-1], Y)
    TPTN = 0.0
    TPFP = 0
    TNFN = 0
    TP = 0
    TN = 0
    # run again the best case
    for i in range(len(XTest)):
        ans = clf.predict([X[i][:-1]])
        if (ans > 0):
            if (ans == X[i][len(X[0]) - 1]):
                TPTN += 1
                TN += 1
            TNFN += 1
        else:
            if (ans == X[i][len(X[0]) - 1]):
                TPTN += 1
                TP += 1
            TPFP += 1
    accuracy = TPTN / (TPFP + TNFN)
    precision = TP / TPFP
    recall = TP / (TP + TNFN - TN)
    print('treating class 2 as positive')
    print('the Best K is ' + str((1)))
    print('Accuracy = ' + str(accuracy))
    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 Measure = ' + str(2 * precision * recall / (precision + recall)))


def generateData2(mean_11,mean_12,mean_13,mean_21,mean_22,mean_23,cov_1,cov_2,cov_3):
    mean11 = CSVfileReader(mean_11)
    mean12 = CSVfileReader(mean_12)
    mean13 = CSVfileReader(mean_13)
    mean21 = CSVfileReader(mean_21)
    mean22 = CSVfileReader(mean_22)
    mean23 = CSVfileReader(mean_23)
    cov1 = CSVfileReader(cov_1)
    cov2 = CSVfileReader(cov_2)
    cov3 = CSVfileReader(cov_3)
    class11 = np.random.multivariate_normal(mean11[0], cov1, 2000)
    class12 = np.random.multivariate_normal(mean12[0], cov2, 2000)
    class13 = np.random.multivariate_normal(mean13[0], cov3, 2000)
    class21 = np.random.multivariate_normal(mean21[0], cov1, 2000)
    class22 = np.random.multivariate_normal(mean22[0], cov2, 2000)
    class23 = np.random.multivariate_normal(mean23[0], cov3, 2000)
    np.random.shuffle(class11)
    np.random.shuffle(class12)
    np.random.shuffle(class13)
    np.random.shuffle(class21)
    np.random.shuffle(class22)
    np.random.shuffle(class23)
    c1 = []
    c2 = []
    counter1 =0
    counter2 = 0
    counter3 =0
    for i in range(2000):
        choice = np.random.choice([1,2,3],1,p=[0.1,0.42,0.48])
        if (choice == 1):
            c1.append(class11[i])
            c2.append(class21[i])
            counter1+=1

        elif (choice == 2):
            c1.append(class12[i])
            c2.append(class22[i])
            counter2+=1
        elif (choice == 3):
            c1.append(class13[i])
            c2.append(class23[i])
            counter3+=1
    dataPool1 = -1 * np.ones((2000, 21), dtype='float')  # -1 when using c1
    dataPool2 = np.ones((2000, 21), dtype='float')  # 1 when using c2
    dataPool1[:, :-1] = c1
    dataPool2[:, :-1] = c2
    np.random.shuffle(dataPool1)
    np.random.shuffle(dataPool2)
    testSet = []
    trainSet = []
    for i in range(2000):
        if (i<600):
            testSet.append(dataPool1[i])
            testSet.append(dataPool2[i])
        else:
            trainSet.append(dataPool1[i])
            trainSet.append(dataPool2[i])
    shuffle(testSet)
    shuffle(trainSet)
    with open('hwk2_datasets_corrected\\DS2_test.csv', 'w', newline='') as file:
        csvWriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for j in range(len(testSet)):
            csvWriter.writerow(testSet[j])
    with open('hwk2_datasets_corrected\\DS2_train.csv', 'w', newline='') as file:
        csvWriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for j in range(len(trainSet)):
            csvWriter.writerow(trainSet[j])





applyKNN('hwk2_datasets_corrected\\DS1_train.csv','hwk2_datasets_corrected\\DS1_test.csv',3)
testKNNSci('hwk2_datasets_corrected\\DS2_train.csv','hwk2_datasets_corrected\\DS2_test.csv')
# generateData2('hwk2_datasets_corrected\\DS2_c1_m1.txt','hwk2_datasets_corrected\\DS2_c1_m2.txt',
#               'hwk2_datasets_corrected\\DS2_c1_m3.txt','hwk2_datasets_corrected\\DS2_c2_m1.txt',
#               'hwk2_datasets_corrected\\DS2_c2_m2.txt','hwk2_datasets_corrected\\DS2_c2_m3.txt',
#               'hwk2_datasets_corrected\\DS2_Cov1.txt','hwk2_datasets_corrected\\DS2_Cov2.txt','hwk2_datasets_corrected\\DS2_Cov3.txt')
#
w = LDA('hwk2_datasets_corrected\\DS2_train.csv')
testLDA('hwk2_datasets_corrected\\DS2_test.csv',w)
