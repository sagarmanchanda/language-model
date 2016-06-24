import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import sys

class RNN:

    def __init__(self, wordDimension, hiddenDimension=100, bpttTruncate=4):
        self.wordDimension = wordDimension
        self.hiddenDimension = hiddenDimension
        self.bpttTruncate = bpttTruncate
        # Giving random initial values to parameters
        self.U = np.random.uniform(-np.sqrt(1.0/wordDimension), np.sqrt(1.0/wordDimension), (hiddenDimension, wordDimension))
        self.V = np.random.uniform(-np.sqrt(1.0/hiddenDimension), np.sqrt(1.0/hiddenDimension), (wordDimension, hiddenDimension))
        self.W = np.random.uniform(-np.sqrt(1.0/hiddenDimension), np.sqrt(1.0/hiddenDimension), (hiddenDimension, hiddenDimension))

    def forwardPropogation(self, x):
        # Total number of time steps, no of words in a sentence
        T = len(x)
        # For forward propogation we would need to store hiddenStates(memory) s for each time T and also initialise s(-1) as 0. Also, we would need to store all the outputs.
        s = np.zeros((T+1, self.hiddenDimension))
        o = np.zeros((T, self.wordDimension))
        # step not required but written for clearity, initializing the s(-1) state as 0
        s[-1] = np.zeros(self.hiddenDimension)
        # forward propogation in play...
        for t in xrange(T):
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = self.V.dot(s[t])
            # Applying softmax, might want to take care of numerical blowup, but wait, tanh is already there, might not be a problem. Think again. For now take the lazy alterative.
            o[t] = np.exp(o[t])
            o[t] = o[t]/np.sum(o[t])
        return [o, s]

    def predict(self, x):
        # for predicting simply forward propogate and return the highest score obtained by the words in each output
        o, s = self.forwardPropogation(x)
        return np.argmax(o, axis=1)

    def calculateLoss(self, x, y):
        loss = 0
        for i in xrange(len(y)):
            o, s = self.forwardPropogation(x[i])
            correctPredictions = o[np.arange(len(y[i])), y[i]]
            loss += -np.sum(np.log(correctPredictions))

        N = np.sum([len(sent) for sent in y])
        return loss/N

    def backPropogationThroughTime(self, x, y):
        T = len(y)
