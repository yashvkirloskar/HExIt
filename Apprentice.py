import numpy as np 
import scipy as sp 
import tensorflow as tf 
import HexitConvNet

class Apprentice:
	def __init__(self, name, board_size, batch_size):
		self.name = name
		self.NN = HexitConvNet.CNN((board_size, board_size, 6), board_size * board_size, 64,batch_size, name)

	def train(self, states, actions, mask):
		self.NN.train(states, actions)

	def getActionDistribution(self, states):
        def get_mask(state):
            return ((state[0, 2:-2, 2:-2] + state[1, 2:-2, 2:-2])-1)*-1
        mask = np.apply_along_axis(get_mask, axis=0)
		return self.NN.predict(state, mask)