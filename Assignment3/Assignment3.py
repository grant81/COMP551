import collections
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def preprocessing(data):
    df = pd.read_csv(data, sep='delimiter', header=None, engine='python')
    df = df[0]
    for i in range(len(df)):
        df[i] = re.sub(r'[^\w]', ' ', str.lower(df[i]))
    return df


def buildVocabulary(dataset, storageLocation):  # 'datasets//yelp-vocab.csv'
    dataset = preprocessing(dataset)
    topWords = collections.Counter()
    for comments in dataset:
        words = comments.split()[:-1]
        topWords.update(words)
    topWords = topWords.most_common(10000)
    Vocabulary = {}
    with open(storageLocation, 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for j in range(len(topWords)):
            csvWriter.writerow([topWords[j][0], j, topWords[j][1]])
            Vocabulary[topWords[j][0]] = ((j, topWords[j][1]))
    return Vocabulary


def buildReview(dataset, vocabulary, storageLocation):
    dataset = preprocessing(dataset)
    reviews = []
    scores = []
    for comment in dataset:
        words = comment.split()
        row = []
        for k in range(len(words) - 1):
            res = vocabulary.get(words[k], -1)
            if (res != -1):
                row.append(res[0])
        if (len(row) > 0):
            reviews.append(row)
            scores.append(words[len(words) - 1])
    with open(storageLocation, 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=' ')
        for j in range(len(reviews)):
            row = []
            for i in range(len(reviews[j])):
                row.append(reviews[j][i])
            row.append('\t' + scores[j])
            csvWriter.writerow(row)
    return (reviews, scores)


def readReviews(reviewPath):
    df = pd.read_csv(reviewPath, sep='\t', names=['review', 'score'], engine='python')
    review = []
    for line in df['review']:
        row = line.split()
        review.append(np.array(row).astype(np.float))
    score = np.array(df['score']).astype(np.float)
    return (review, score)


def readVocab(vocabPath):
    df = pd.read_csv(vocabPath, delimiter='\t', header=None, engine='python')
    return (df[0])


def getBinaryBagOfWord(reviews):
    row = []
    col = []
    value = []
    length = len(reviews)
    for i in range(len(reviews)):
        temp = set()
        for j in range(len(reviews[i])):
            if (reviews[i][j] not in temp):
                temp.add(reviews[i][j])
                col.append(reviews[i][j])
                row.append(i)
                value.append(1)
    BBWMatrix = sparse.bsr_matrix((value, (row, col)))
    if (BBWMatrix.shape[1] != 10000):
        BBWMatrix = sparse.bsr_matrix((BBWMatrix.data, BBWMatrix.indices, BBWMatrix.indptr), shape=(length, 10000))
    # BBWMatrix = BBWMatrix.toarray()
    return BBWMatrix


def getFreqBagOfWord(reviews):
    row = []
    col = []
    value = []
    for i in range(len(reviews)):
        temp = []
        tempValue = []
        for j in range(len(reviews[i])):
            if (reviews[i][j] not in temp):
                temp.append(reviews[i][j])
                col.append(reviews[i][j])
                row.append(i)
                tempValue.append(1 / len(reviews[i]))
            else:
                k = temp.index(reviews[i][j])
                tempValue[k] += 1 / len(reviews[i])
        value += tempValue
    FBWMatrix = sparse.bsr_matrix((value, (row, col)))
    if (FBWMatrix.shape[1] != 10000):
        FBWMatrix = sparse.bsr_matrix((FBWMatrix.data, FBWMatrix.indices, FBWMatrix.indptr),
                                      shape=(len(reviews), 10000))
    # FBWMatrix = FBWMatrix.toarray()
    return FBWMatrix


def NativeBayes(trainingX, trainingY, testingX, testingY):
    clf = GaussianNB()
    clf.fit(trainingX, trainingY)
    preditY = []
    hit = 0
    for i in range(len(testingX)):
        ans = clf.predict([testingX[i]])
        if ans == testingY[i]:
            hit += 1
        preditY.append(ans)

    print('F1 weighted = ' + str(metrics.f1_score(testingY, preditY, average='weighted')))


def BernoulliNativeBayes(trainingX, trainingY, testingX, testingY, a):
    clf = BernoulliNB(alpha=a)
    clf.fit(trainingX, trainingY)
    preditY = []
    hit = 0
    for i in range(len(testingX)):
        ans = clf.predict([testingX[i]])
        if ans == testingY[i]:
            hit += 1
        preditY.append(ans)
    f1 = metrics.f1_score(testingY, preditY, average='weighted')
    # print('F1 Score = ' + str(f1))
    return f1


def majorityClassifier(trainingX, trainingY, testingX, testingY):
    clfMajority = DummyClassifier(strategy='most_frequent', random_state=0)
    clfMajority.fit(trainingX, trainingY)
    preditMajority = []
    hit = 0
    for i in range(len(testingX)):
        ans2 = clfMajority.predict([testingX[i]])
        preditMajority.append(ans2)
    print('Most Frequent F1 Score= ' + str(metrics.f1_score(testingY, preditMajority, average='weighted')))


def randomClassifier(trainingX, trainingY, testingX, testingY):
    clfUniform = DummyClassifier(strategy='uniform', random_state=0)
    clfUniform.fit(trainingX, trainingY)
    preditYUniform = []
    for i in range(len(testingX)):
        ans1 = clfUniform.predict([testingX[i]])
        preditYUniform.append(ans1)
    print('Uniform Random F1 Score= ' + str(metrics.f1_score(testingY, preditYUniform, average='weighted')))


def LinearSVM(trainingX, trainingY, testingX, testingY, C):
    clf = LinearSVC(C=C)
    clf.fit(trainingX, trainingY)
    preditY = []
    for i in range(len(testingX)):
        ans = clf.predict([testingX[i]])
        preditY.append(ans)
    f1Score = metrics.f1_score(testingY, preditY, average='weighted')
    return f1Score


def decisionTree(trainingX, trainingY, testingX, testingY, maxLeaf):
    clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=maxLeaf)
    clf.fit(trainingX, trainingY)
    preditY = []
    for i in range(len(testingX)):
        ans = clf.predict([testingX[i]])
        preditY.append(ans)
    f1Score = metrics.f1_score(testingY, preditY, average='weighted')
    return f1Score


def plotHyperparameterTrainingProgress(input):
    a = np.array(input, dtype='float')
    a = np.transpose(a)
    plt.plot(a[1], a[0], 'bo')
    plt.title('Selecting HyperParameter')
    plt.xlabel('Hyperparameter')
    plt.ylabel('F1 Score')
    plt.show()


trainResult = readReviews('outputdatasets//yelp-train.txt')
trainX = getBinaryBagOfWord(trainResult[0])
trainY = trainResult[1]
testResult = readReviews('outputdatasets//yelp-valid.txt')
testX = getBinaryBagOfWord(testResult[0]).toarray()
testY = testResult[1]
# LinearSVM(trainX,trainY,testX,testY)
# BernoulliNativeBayes(trainX,trainY,testX,testY)
# decisionTree(trainX, trainY, testX, testY, 30)
# majorityClassifier(trainX,trainY,testX,testY)
# randomClassifier(trainX,trainY,testX,testY)

f1s = []
# for i in range (500,100,-10):
#     print('depth of the tree = '+str(i))
#     f1s.append([decisionTree(trainX,trainY,testX,testY,i),i])
# f1s.sort()
# print('the optimal number of leafs is '+f1s[0][1]+', it achieves a F1 score of '+ f1s[0][0])
# a = 5
# for i in range(0,30):
#
#     a = 0.5**i
#     # print('alpha = '+str(a))
#     f1s.append([BernoulliNativeBayes(trainX, trainY, testX, testY, a), a])
# f1s.sort()
# plotHyperparameterTrainingProgress(f1s)
# print('the optimal alpha is ' + str(f1s[len(f1s)-1][1]) + ', it achieves a F1 score of ' + str(f1s[len(f1s)-1][0]))

C = 1
for i in range(0, 30):
    C = C * 0.5 ** i
    # print('alpha = '+str(a))
    f1s.append([LinearSVM(trainX, trainY, testX, testY, C), C])
f1s.sort()
plotHyperparameterTrainingProgress(f1s)
print('the optimal C is ' + str(f1s[len(f1s) - 1][1]) + ', it achieves a F1 score of ' + str(f1s[len(f1s) - 1][0]))
