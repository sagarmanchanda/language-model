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
        o, s = self.forwardPropogation(x)
        dlossdU = np.zeros(self.U.shape)
        dlossdW = np.zeros(self.W.shape)
        dlossdV = np.zeros(self.V.shape)
        dOutput = o
        dOutput[np.arange(T), y] -= 1.

        for t in range(T-1,-1,-1):
            # s[t].T is the transpose
            dlossdV += np.outer(dOutput[t], s[t].T)
            dt = self.V.T.dot(dOutput[t]) * (1-(s[t]**2))
            #now we backpropogate through time for self.bpttTruncate units
            for bpptStep in range(t, max(-1, t-self.bpttTruncate-1), -1):
                dlossdW += np.outer(dt, s[bpptStep-1])
                dlossdU[:,x[bpptStep]] += dt
                dt = self.W.T.dot(dt) * (1 - (s[bpptStep-1] ** 2))

        return [dlossdU, dlossdV, dlossdW]


    def sgdStep(self, x, y, learningRate):
        dlossdU, dlossdV, dlossdW = self.backPropogationThroughTime(x, y)
        self.U -= learningRate * dlossdU
        self.V -= learningRate * dlossdV
        self.W -= learningRate * dlossdW

    def trainModel(self, X_train, y_train, learningRate, nepoch=100, evaluateLossAfter=5):
        losses = []
        examplesSeen = 0
        for epoch in xrange(nepoch):
            if (epoch%evaluateLossAfter == 0):
                loss = self.calculateLoss(X_train, y_train)
                losses.append((examplesSeen, loss))
                print "Loss after seeing %d examples and %d epoch is %f" % (examplesSeen, epoch, loss)
                # Need to decrease the learning rate if loss starts increasing
                if (len(losses)>1 and losses[-1][1] > losses[-2][1]):
                    learningRate *= 0.5
                    print "Learning rate updated to %f" % (learningRate)
                sys.stdout.flush()

            for i in range(len(y_train)):
                print "Epoch: %d Example: %d" % (epoch, i)
                self.sgdStep(X_train[i], y_train[i], learningRate)
                examplesSeen += 1

            print "Saving the value of parameters for current epoch"
            u_outfile = open('saved-states/'+str(epoch)+'_u.npy','w')
            v_outfile = open('saved-states/'+str(epoch)+'_v.npy','w')
            w_outfile = open('saved-states/'+str(epoch)+'_w.npy','w')
            np.save(u_outfile, self.U)
            np.save(v_outfile, self.V)
            np.save(w_outfile, self.W)