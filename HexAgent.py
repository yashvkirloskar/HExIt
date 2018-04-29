from Apprentice import *
from Expert import *

class HexAgent:
    def __init__(self, name, board_size=5, batch_size=256, simulations_per_state=1000, max_depth=6):
        self.apprentice = Apprentice(board_size, batch_size, simulations_per_state, max_depth)
        self.expert = Expert(name, board_size, batch_size)
        self.addApprentice = True

    def train(self):
        batch, distributions = self.expert.generateBatch()
        self.apprentice.train(batch, distributions)
        if self.addApprentice:
            self.expert = Expert(name, board_size, batch_size, self.apprentice)
            self.addApprentice = False

    def getMove(self, state):
        return self.expert.getMove(state)