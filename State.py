import numpy as np
import Hex

class State:
	def __init__(self, game_size):
		self.game_size = game_size
	
    def __eq__(self, other):
		
	
    def __repr__(self):
		

    def winner(self):
        pass

    def isTerminalState(self):
        pass

    def calculateReward(self):
        pass

    def nextState(self, action):
        pass

    def isLegalAction(self, action):
        pass

    def legalActions(self):
        pass

    def chooseRandomAction(self):
        pass

    def turn(self):
        pass

    def isPlayerOneTurn(self):
        pass

    def isPlayerTwoTurn(self):
        pass

    def nonTerminalActions(self):
        pass
