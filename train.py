import numpy as np
import rnn
import time

X_train = np.load('saved-states/X_train.npy')
y_train = np.load('saved-states/y_train.npy')

model = rnn.RNN(hiddenDimension=50, wordDimension=8000, bpttTruncate=4)
model.trainModel(X_train[:10000], y_train[:10000], learningRate=0.005, nepoch=30, evaluateLossAfter=1)