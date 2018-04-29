import numpy as np 
import scipy as sp 
import tensorflow as tf 
import HexitConvNet

class Apprentice:
    def __init__(self, name, board_size, batch_size):
        self.batch_size = batch_size
        self.board_size = board_size
        self.name = name
        self.NN = HexitConvNet.CNN((6, board_size+4, board_size+4), board_size * board_size, 64,batch_size, name)

    def train(self, states, actions, mask=None):
        print (states.shape)
        mask = np.zeros((states.shape[0], self.board_size*self.board_size))
        for i, state in enumerate(states):
            mask[i] = (((state[0, 2:-2, 2:-2] + state[1, 2:-2, 2:-2])-1)*-1).flatten()
        self.NN.train(states, actions, mask)

    def getActionDistribution(self, states, mask=None):
        mask = np.zeros((states.shape[0], self.board_size*self.board_size))
        for i, state in enumerate(states):
            mask[i] = (((state[0, 2:-2, 2:-2] + state[1, 2:-2, 2:-2])-1)*-1).flatten()
        return self.NN.predict(state, mask)