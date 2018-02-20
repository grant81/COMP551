import collections
import csv
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
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


def buildReview(dataset, vocabulary,storageLocation):
    dataset = preprocessing(dataset)
    reviews = []
    scores = []
    for comment in dataset:
        words = comment.split()
        row = []
        for k in range(len(words)-1):
            res = vocabulary.get(words[k],-1)
            if (res != -1):
                row.append(res[0])
        if(len(row)>0):
            reviews.append(row)
            scores.append(words[len(words)-1])
    with open(storageLocation, 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=' ')
        for j in range(len(reviews)):
            row = []
            for i in range(len(reviews[j])):
                row.append(reviews[j][i])
            row.append('\t'+scores[j])
            csvWriter.writerow(row)
    return (reviews,scores)


def readReviews(reviewPath):
    df = pd.read_csv(reviewPath, sep='\t',names=['review','score'], engine='python')
    review = []
    for line in df['review']:
        row = line.split()
        review.append(np.array(row).astype(np.float))
    score = np.array(df['score']).astype(np.float)
    return (review,score)


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
            if(reviews[i][j] not in temp):
                temp.add(reviews[i][j])
                col.append(reviews[i][j])
                row.append(i)
                value.append(1)
    BBWMatrix = sparse.bsr_matrix((value, (row, col)), shape=(length, 10000)).toarray()
    return BBWMatrix


def getFreqBagOfWord(reviews):
    row = []
    col = []
    value = []
    for i in range(len(reviews)):
        temp = []
        tempValue = []
        for j in range(len(reviews[i])):
            if(reviews[i][j] not in temp):
                temp.append(reviews[i][j])
                col.append(reviews[i][j])
                row.append(i)
                tempValue.append(1/len(reviews[i]))
            else:
                k = temp.index(reviews[i][j])
                tempValue[k]+=1/len(reviews[i])
        value += tempValue
    FBWMatrix = sparse.bsr_matrix((value, (row, col)), shape=(len(reviews), 10000)).toarray()
    return FBWMatrix


def NativeBayes(trainingX,trainingY,testingX,testingY):
    clf = GaussianNB()
    clf.fit(trainingX, trainingY)
    GaussianNB(priors=None)
    preditY = []
    hit = 0
    for i in range(len(testingX)):
        ans = clf.predict([testingX[i]])
        if ans == testingY[i]:
            hit +=1
        preditY.append(ans)

    # print('accrucy = ' + str(hit/len(testingY)))
    print('accrucy = ' + str(metrics.accuracy_score(testingY,preditY)))

# vocab = readVocab('outputdatasets//yelp-vocab.txt')
trainResult = readReviews('outputdatasets//IMDB-train.txt')
trainX = getFreqBagOfWord(trainResult[0])
trainY = trainResult[1]
testResult = readReviews('outputdatasets//IMDB-train.txt')
testX = getFreqBagOfWord(testResult[0])
testY = testResult[1]
NativeBayes(trainX,trainY,testX,testY)

# vocab = buildVocabulary('datasets//IMDB-train.txt','outputdatasets//IMDB-vocab.txt')
# trainResult = buildReview('datasets//IMDB-train.txt',vocab,'outputdatasets//IMDB-train.txt')
# trainX = getFreqBagOfWord(trainResult[0])
# trainY = trainResult[1]
# testResult = buildReview('datasets//IMDB-train.txt',vocab,'outputdatasets//IMDB-train.txt')
# testX = getFreqBagOfWord(testResult[0])
# testY = testResult[1]
# NativeBayes(trainX,trainY,testX,testY)
# buildReview('datasets//yelp-test.txt',vocab,'outputdatasets//yelp-test.txt')
# buildReview('datasets//yelp-valid.txt',vocab,'outputdatasets//yelp-valid.txt')
# vocab = buildVocabulary('datasets//IMDB-train.txt','outputdatasets//IMDB-vocab.txt')
# buildReview('datasets//IMDB-train.txt',vocab,'outputdatasets//IMDB-train.txt')
# buildReview('datasets//IMDB-test.txt',vocab,'outputdatasets//IMDB-test.txt')
# buildReview('datasets//IMDB-valid.txt',vocab,'outputdatasets//IMDB-valid.txt')
#
# print('done')
