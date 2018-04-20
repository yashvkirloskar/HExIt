import util
import numpy as np
from State import State
from MCTS import MCTS
from MCTS_Tree import MCTS_Tree, MCTS_Node
import random, math
from scipy.special import comb

# to test, we will use a tic tac toe board
class TestState:
	def __init__(self, squares):
		self.squares = squares
	def __eq__(self, other):
		return self.squares == other.squares
	def __repr__(self):
		s = "\n"
		for i in range(3):
			for j in range(3):
				val = self.squares[i][j]
				s += "X " if val == 1 else ("O " if val == -1 else "_ ")
			s += "\n"
		return s

	def winner(self):
		t = self.squares
		# check each row
		for i in range(3):
			if np.all(t[i] == [1,1,1]):
				return 1
			if np.all(t[i] == [-1,-1,-1]):
				return -1

		# check each col
		for j in range(3):
			if np.all(t[:,j] == [1,1,1]):
				return 1
			if np.all(t[:,j] == [-1,-1,-1]):
				return -1

		# check top left bottom right diag
		if t[0,0] == t[1,1] == t[2,2] == 1:
			return 1
		if t[0,0] == t[1,1] == t[2,2] == -1:
			return -1

		# check top right bottom left diag
		if t[0,2] == t[1,1] == t[2,0] == 1:
			return 1
		if t[0,2] == t[1,1] == t[2,0] == -1:
			return -1

		return 0

	def isTerminalState(self):
		w = self.winner()
		if w == 1 or w == -1:
			return True
		# check if draw
		if np.all(self.squares != 0):
			return True

		return False

	def calculateReward(self):
		return self.winner()

	def nextState(self, action):
		# Assume given action is legal
		row, col = action // 3, action % 3
		turn = 1 if np.sum(self.squares) == 0 else -1
		new_squares = self.squares.copy()
		new_squares[row,col] = turn
		return TestState(new_squares)

	def isLegalAction(self, action):
		row, col = action // 3, action % 3
		return self.squares[row,col] == 0

	def legalActions(self):
		la = [a for a in range(9) if self.isLegalAction(a)]
		return la

	def chooseRandomAction(self):
		if len(self.legalActions()) == 0:
			return -1 # this really shouldn't happen
		return random.choice(self.legalActions())

	def turn(self):
		return 1 if np.sum(self.squares) == 0 else -1

	def isPlayerOneTurn(self):
		return self.turn() == 1

	def isPlayerTwoTurn(self):
		return self.turn() == -1

	def nonTerminalActions(self):
		la = self.legalActions()
		nta = []
		for action in la:
			b = self.nextState(action)
			if not b.isTerminalState():
				nta.append(action)
		return nta


def perm(n, k):
	return math.factorial(n) / math.factorial(k)

def get_config_proportions():
	counts = np.zeros(5)
	for i in range(5):
		if i == 0:
			counts[0] = 10
		else:
			counts[i] = (comb(9, i*2) * comb(i*2, i)) + (comb(9, i*2 + 1) * comb(i*2 + 1, i))
	return counts/np.sum(counts)

# Generates a random non terminal state
# Currently, not totally random.  The choice of num_squares_per has to be proportional to the number of possible configs.
# But, we'll fix this later.
def generateRandomTestState():
	turn = random.choice([1,-1])
	num_squares_per = np.random.choice(5, p=get_config_proportions())
	if num_squares_per == 4:
		turn = 1

	board = TestState(np.zeros(9).reshape(3,3))
	for i in range(num_squares_per):
		p1_options = board.nonTerminalActions()
		if len(p1_options) == 0:
			return board
		p1_action = random.choice(p1_options)
		board = board.nextState(p1_action)

		p2_options = board.nonTerminalActions()
		if len(p2_options) == 0:
			return board
		p2_action = random.choice(p2_options)
		board = board.nextState(p2_action)

	if turn == -1:
		p1_options = board.nonTerminalActions()
		if len(p1_options) == 0:
			return board
		p1_action = random.choice(p1_options)
		board = board.nextState(p1_action)

	return board


		


def main():
	squares = np.array(
		[[1,1,-1],
		 [-1,0,-1],
		 [1,0,1]])
	t = TestState(squares)

	start_states = [generateRandomTestState() for i in range(16)]
	m = MCTS(num_actions=9, batch_size=16, simulations_per_state=1000)
	ss, p1, p2 = m.generateDataBatch(start_states)
	for i in range(16):
		print(ss[i])
		print(p1[i])
		print(p2[i])
	#states = [State(x) for x in np.random.randint(0, 1000, 16)]
	#data = m.generateDataBatch(states)


if __name__ == '__main__':
	main()