import util
import numpy as np
from State import State, TestState
from MCTS import MCTS
from MCTS_Tree import MCTS_Tree, MCTS_Node
import random, math
from scipy.special import comb


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

def action_string(action):
	if action == 9:
		return "dummy"
	row, col = action // 3, action % 3
	row_string = "top" if row == 0 else ("middle" if row == 1 else "bottom")
	col_string = "left" if col == 0 else ("center" if col == 1 else "right")
	s = row_string + ", " + col_string
	return s
		


def main():
	squares = np.array(
		[[-1,1,-1],
		 [1,1,0],
		 [0,-1,0]])
	t = TestState(squares)

	start_states = [generateRandomTestState() for i in range(16)]
	m = MCTS(num_actions=9, batch_size=16, simulations_per_state=1000)
	ss, p1, p2 = m.generateDataBatch(start_states)
	for i in range(16):
		print("BOARD:")
		print(ss[i])
		if ss[i].isPlayerOneTurn():
			print("X's turn")
			print(action_string(np.argmax(p1[i])))
			print(np.array(p1[i][0:9]).reshape(3,3))
		else:
			print("O's turn")
			print(action_string(np.argmax(p2[i])))
			print(np.array(p2[i][0:9]).reshape(3,3))
		


if __name__ == '__main__':
	main()