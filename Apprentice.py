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
        print("Entered Apprentice Train Function")
        mask = np.zeros((states.shape[0], self.board_size*self.board_size))
        for i, state in enumerate(states):
            mask[i] = (((state[0, 2:-2, 2:-2] + state[1, 2:-2, 2:-2])-1)*-1).flatten()
        self.NN.train(states, actions, mask)
        print("Exited Apprentice Train Functions")

    def getActionDistribution(self, states, mask=None):
        mask = np.zeros((states.shape[0], self.board_size*self.board_size))
        for i, state in enumerate(states):
            mask[i] = (((state[0, 2:-2, 2:-2] + state[1, 2:-2, 2:-2])-1)*-1).flatten()
        return self.NN.predict(states, mask)

    def predict(self, states, mask=None):
        print("Entered Apprentice Predict Function")
        return self.getActionDistribution(states, mask)
        print("Exited Apprentice Predict Function")

    # Takes in a state_to_channel of size [1, 6, 9, 9] and outputs a [25,] action distribution 
    def getActionDistributionSingle(self, state, turn, mask=None):
        stateInput = np.zeros((2, 6, self.board_size+4, self.board_size+4))
        stateInput[0 if turn == 1 else 1] = state
        mask = np.zeros((2, self.board_size*self.board_size))
        mask[0 if turn == 1 else 1] = (((state[0, 2:-2, 2:-2] + state[1, 2:-2, 2:-2])-1)*-1).flatten()
        return self.NN.predict(stateInput, mask)[0 if turn == 1 else 1]
