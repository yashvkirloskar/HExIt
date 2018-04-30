import numpy as np 
from MCTS import MCTS

class Expert:

	def __init__(self, name, board_size=5, batch_size=256, simulations_per_state=1000, max_depth=6, apprentice=None):
		self.name = name
		self.board_size = board_size
		self.batch_size = batch_size
		self.simulations_per_state = simulations_per_state
		self.max_depth = max_depth
		self.apprentice = apprentice

		self.mcts = MCTS(size=board_size, batch_size=batch_size, simulations_per_state=simulations_per_state, max_depth=max_depth, apprentice=apprentice)


	def generateBatch(self):
		return self.mcts.generateExpertBatch(outFile1=None, outFile2=None)


	def getMove(self, state):
		print ("Getting move from expert")
		return self.mcts.getMove(state)
		
