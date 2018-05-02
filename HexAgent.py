from Apprentice import *
from Expert import *
import os
import numpy as np

class HexAgent:
    def __init__(self, name, board_size=5, batch_size=256, simulations_per_state=1000, max_depth=6):
        self.name = name
        self.board_size = board_size
        self.batch_size = batch_size
        self.simulations_per_state = simulations_per_state
        self.max_depth = max_depth

        self.apprentice = Apprentice(name=name, board_size=board_size, batch_size=batch_size)
        if(os.path.exists(name)):
            self.expert = Expert(board_size=board_size, batch_size=batch_size, simulations_per_state=simulations_per_state, max_depth=max_depth, apprentice=self.apprentice)
            self.addApprentice = False
        else:
            self.expert = Expert(board_size=board_size, batch_size=batch_size, simulations_per_state=simulations_per_state, max_depth=max_depth, apprentice=None)
            self.addApprentice = True

    def train(self, samples):
        for i in range(samples):
            batch, distributions = self.expert.generateBatch()
            np.save(str(i)+"batch", batch)
            np.save(str(i)+"distributions", distributions)
        for i in range(samples):
            batch = np.load(str(i)+"batch.npy")
            distributions = np.load(str(i)+"distributions.npy")
            self.apprentice.train(batch, distributions)
        if self.addApprentice:
            self.expert = Expert(board_size=self.board_size, batch_size=self.batch_size, simulations_per_state=self.simulations_per_state, max_depth=self.max_depth, apprentice=self.apprentice)
            self.addApprentice = False

    # Do not call this method unless you have called train
    def getMove(self, state):
        print ("Getting move from AI")
        if self.addApprentice:
            self.expert = Expert(board_size=self.board_size, batch_size=self.batch_size, simulations_per_state=self.simulations_per_state, max_depth=self.max_depth, apprentice=self.apprentice)
        return self.expert.getMove(state)

    def getRandomMove(self, state):
        return state.chooseRandomAction()