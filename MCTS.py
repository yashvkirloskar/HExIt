import util
import numpy as np
from MCTS_Tree import MCTS_Tree, MCTS_Node
from State import State
from MCTS_utils import *
import time

class MCTS:
	#uct_new(s,a) = uct(s,a) + w_a * (apprentice_policy(s,a) / n(s,a) + 1)
	# given a list of batch_size state objects
	# return an array with batch_size elements.  Each element is a 26-list.  The state, followed by number of times we took action i
	#uct(s,a) = r(s,a)/n(s,a) + c_b * sqrt ( log(n(s)) / n(s,a) )
	
	#rollout begind at state s' we've never seen before. finish sim, add s' to tree. propagate signal up 
	def __init__(self, num_actions=5, batch_size=256, simulations_per_state=1000, max_depth=6, apprentice=None):
		print ("initialized MCTS")
		self.size = num_actions
		self.num_actions = num_actions**2
		self.batch_size = batch_size
		self.simulations_per_state = simulations_per_state
		self.max_depth = max_depth
		self.apprentice = apprentice

	# This method generates a dataset of size SELF.BATCH_SIZE.
	# It is passed STARTING_STATES, which is a list of BATCH_SIZE states (as State instances).
	# For each starting state, runs SELF.SIMULATIONS_PER_STATE, each starting at that start state.
	# Calculates the number of times each action was taken from the root node (start state).
	# Returns three arrays, S, A1 and A2, each with BATCH_SIZE + 1 elements.
	# S is just a copy of STARTING_STATES.
	# The i-th element of A gives the distribution of actions from the i-th start state for Player 1.
	# The i-th element of A gives the distribution of actions from the i-th start state for Player 2.
	# If it is Player X's turn, then Player Y's distribution will be all 0's except for the last position is a 1.
	# This data is to be passed to the apprentice, that will be trained to mimic this distribution.
	def generateDataBatch(self, starting_states):

		action_distribution1 = np.zeros(shape=(self.batch_size, self.num_actions + 1))
		action_distribution2 = np.zeros(shape=(self.batch_size, self.num_actions + 1))

		# run all the starting states through the apprentice as once
		root_action_distributions = [None for i in range(self.num_actions)]
		if self.apprentice is not None:
			root_action_distributions = self.apprentice.getActionDistribution(starting_states)
			# this is a [batch_size, num_actions] shaped matrix

		for i, state in enumerate(starting_states):
			print("i:", i)
			if state.isPlayerOneTurn():
				action_distribution1[i][0:self.num_actions] = self.runSimulations(state, root_action_distributions[i])
				action_distribution2[i][-1] = 1
			else:
				action_distribution2[i][0:self.num_actions] = self.runSimulations(state, root_action_distributions[i])
				action_distribution1[i][-1] = 1

		return (starting_states, action_distribution1, action_distribution2)


	# Runs SIMULATIONS_PER_STATE simulations, each starting from the given START_STATE.
	# Returns a list with as many elements as there are actions, plus 1. (ex. 26 for a 5x5 hex game).
	# Each element is a probability (between 0 and 1).
	# The i-th element is the number of times we took the i-th action from the root state (as a probability).
	# The last element is the number of times we took no action (if it wasn't this player's turn.)
	def runSimulations(self, start_state, root_action_distribution):

		# Initialize new tree
		self.tree = MCTS_Tree(start_state, self.num_actions, root_action_distribution=root_action_distribution, max_depth=self.max_depth, apprentice=self.apprentice)
		for t in range(self.simulations_per_state):
			self.tree.runSingleSimulation()

		return self.tree.getActionCounts() / self.simulations_per_state

	# Returns a np array of shape [2, batch_size, 6, 5, 5] of input data for white and black
	# and a np array of shape [2, batch_size, 25] for white and black
	# Takes in one output for distributions and one for inputs
	def generateExpertBatch(self, outFile1=None, outFile2=None):
		p1States = []
		p2States = []
		# Generate BATCH_SIZE number of random states for each player
		start = time.time()
		for i in range(self.batch_size):
			p1States.append(generateRandomState(self.size, 1))
		for i in range(self.batch_size):
			p2States.append(generateRandomState(self.size, -1))
		end = time.time()
		print ("Time to generate 512 random states: ", (end-start))

		start = time.time()
		# We don't care about P2's action distribution, and the last column of -1's is unnecessary  
		p1DataBatch = self.generateDataBatch(p1States)[0:2]
		p1Dist = p1DataBatch[1][:, :-1]


		p2DataBatch = self.generateDataBatch(p2States)[0:3:2]
		p2Dist = p2DataBatch[1][:, :-1]


		input_data = np.zeros((2, self.batch_size, 6, self.size+4, self.size+4))

		distributions = np.zeros((2, self.batch_size, self.num_actions))
		distributions[0] = p1Dist
		distributions[1] = p2Dist
		end = time.time()
		print ("Time to do MCTS: ", (end-start))


		start = time.time()
		for i in range(self.batch_size):
			input_data[0, i] = p1DataBatch[0][i].channels_from_state()

		for i in range(self.batch_size):
			input_data[1, i] = p2DataBatch[0][i].channels_from_state()
		if outFile1 is not None and outFile2 is not None:
			np.save(outFile1, input_data)
			np.save(outFile2, distributions)
		end = time.time()
		print ("Time to create inputs for apprentice: ", (end-start))
		return (input_data, distributions)


