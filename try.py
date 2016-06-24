import numpy as np
import rnn

X_train = np.load('saved-states/X_train.npy')
y_train = np.load('saved-states/y_train.npy')

model = rnn.RNN(8000)
print model.calculateLoss(X_train[:1000], y_train[:1000])