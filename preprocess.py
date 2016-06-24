import numpy as np
import matplotlib.pyplot as plt
import nltk
from datetime import datetime
import itertools
import operator
import csv
import sys

# Reading data and pre-processing

vocabularySize = 8000
unknownToken = "UNKNOWN"
sentenceStartToken = "SENTENCE_START"
sentenceEndToken = "SENTENCE_END"


with open('dataset/reddit-comments.csv') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    sentences = ["%s %s %s" % (sentenceStartToken, x, sentenceEndToken) for x in sentences]

tokenizedSentences = [nltk.word_tokenize(x) for x in sentences]
wordFrequency = nltk.FreqDist(itertools.chain(*tokenizedSentences))

# print len(wordFrequency.items())

vocabulary = wordFrequency.most_common(vocabularySize-1)
indexToWord = [x[0] for x in vocabulary]
indexToWord.append(unknownToken)
wordToIndex = dict([(w,i) for i, w in enumerate(indexToWord)])
 
# print wordToIndex

for i, sentence in enumerate(tokenizedSentences):
    tokenizedSentences[i] = [w if w in wordToIndex else unknownToken for w in sentence]

# print tokenizedSentences[0]
# print wordToIndex["SENTENCE_START"]

X_train = np.asarray([[wordToIndex[w] for w in sent[:-1]] for sent in tokenizedSentences])
y_train = np.asarray([[wordToIndex[w] for w in sent[1:]] for sent in tokenizedSentences])


# Saving the preprocessed arrays into files
X_outfile = open('saved-states/X_train.npy', 'w')
y_outfile = open('saved-states/y_train.npy', 'w')
indexToWord_outfile = open('saved-states/indexToWord.npy', 'w')
wordToIndex_outfile = open('saved-states/wordToIndex.npy', 'w')
np.save(X_outfile, X_train)
np.save(y_outfile, y_train)
np.save(indexToWord_outfile, indexToWord)
np.save(wordToIndex_outfile, wordToIndex)