import util
import numpy as np
from MCTS_Tree import MCTS_Tree, MCTS_Node
from State import State
from MCTS_utils import *
import time
import multiprocessing, logging
from multiprocessing import Pool

class MCTS:
	#uct_new(s,a) = uct(s,a) + w_a * (apprentice_policy(s,a) / n(s,a) + 1)
	# given a list of batch_size state objects
	# return an array with batch_size elements.  Each element is a 26-list.  The state, followed by number of times we took action i
	#uct(s,a) = r(s,a)/n(s,a) + c_b * sqrt ( log(n(s)) / n(s,a) )
	
	#rollout begind at state s' we've never seen before. finish sim, add s' to tree. propagate signal up 
	def __init__(self, size=5, batch_size=256, simulations_per_state=1000, max_depth=6, apprentice=None, parallel=False):
		self.size = size
		self.num_actions = size**2
		self.batch_size = batch_size
		self.simulations_per_state = simulations_per_state
		self.max_depth = max_depth
		self.apprentice = apprentice
		self.parallel = parallel


	# Runs SIMULATIONS_PER_STATE simulations on the given state, 
	# collects the action distribution, and returns the argmax action.
	def getMove(self, state):
		# run all the starting states through the apprentice once, for efficiency
		root_action_distribution = [None for i in range(self.num_actions)]
		if self.apprentice is not None:
			state_input = state.channels_from_state()
			root_action_distribution = self.apprentice.getActionDistributionSingle(state_input, state.turn())
			# this is a [num_actions,] shaped vector

		action_distribution = self.runSimulations(state, root_action_distribution)
		return np.argmax(action_distribution)


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
	def generateDataBatch(self, starting_states, starting_inputs):
		action_distribution1 = np.zeros(shape=(self.batch_size, self.num_actions + 1))
		action_distribution2 = np.zeros(shape=(self.batch_size, self.num_actions + 1))

		# run all the starting states through the apprentice as once
		#root_action_distributions = [None for i in range(self.num_actions)]
		root_action_distributions = np.zeros((2, self.batch_size, self.num_actions))
		if self.apprentice is not None:
			root_action_distributions = self.apprentice.getActionDistribution(starting_inputs)
			print("root_action_distributions shape:", root_action_distributions.shape)
			# this is a [batch_size, num_actions] shaped matrix

		for i, state in enumerate(starting_states):
			print("i:", i)
			if state.isPlayerOneTurn():
				action_distribution1[i][0:self.num_actions] = self.runSimulations(state, root_action_distributions[0][i])
				action_distribution2[i][-1] = 1
			else:
				action_distribution2[i-self.batch_size][0:self.num_actions] = self.runSimulations(state, root_action_distributions[1][i - self.batch_size])
				action_distribution1[i-self.batch_size][-1] = 1

		return (starting_states, action_distribution1, action_distribution2)

	def parallelGenerateDataBatch(self, starting_states, starting_inputs):
		action_distribution1 = np.zeros(shape=(self.batch_size, self.num_actions + 1))
		action_distribution2 = np.zeros(shape=(self.batch_size, self.num_actions + 1))

		# run all the starting states through the apprentice as once
		#root_action_distributions = [None for i in range(self.num_actions)]
		# root_action_distributions = np.zeros((2, self.batch_size, self.num_actions))
		# if self.apprentice is not None:
		# 	root_action_distributions = self.apprentice.getActionDistribution(starting_inputs)
		# 	print("root_action_distributions shape:", root_action_distributions.shape)
			# this is a [batch_size, num_actions] shaped matrix
		# map_input = np.array([(starting_states[i], root_action_distributions[0 if i < self.batch_size else 1][i % self.batch_size]) for i in range(2*self.batch_size)])
		# print (map_input.shape)
		logger = multiprocessing.log_to_stderr()
		logger.setLevel(logging.INFO)
		p = Pool()
		distributions = np.array(p.map(func=self.parallelRunSimulations, iterable=starting_states))
		p.close()
		p.join()
		print (distributions.shape)
		action_distribution1 = distributions[0:self.batch_size]
		action_distribution2 = distributions[self.batch_size:]
		return (startingStates, action_distribution1, action_distribution2)


	def parallelRunSimulations(self, state):
		root_action_distribution = None
		print (state.board)
		print (multiprocessing.current_process())
		if state.isPlayerOneTurn():
			action_distribution1[i][0:self.num_actions] = self.runSimulations(state, root_action_distribution)
			action_distribution2[i][-1] = 1
		else:
			action_distribution2[i-self.batch_size][0:self.num_actions] = self.runSimulations(state, root_action_distribution)
			action_distribution1[i-self.batch_size][-1] = 1
		return np.array([action_distribution1, action_distribution2])

	# Runs SIMULATIONS_PER_STATE simulations, each starting from the given START_STATE.
	# Returns a list with as many elements as there are actions, plus 1. (ex. 26 for a 5x5 hex game).
	# Each element is a probability (between 0 and 1).
	# The i-th element is the number of times we took the i-th action from the root state (as a probability).
	# The last element is the number of times we took no action (if it wasn't this player's turn.)
	def runSimulations(self, start_state, root_action_distribution):
		if self.apprentice is not None and root_action_distribution is not None:
			print("in runSimulations, root_action_distribution shape:", root_action_distribution.shape)
		# Initialize new tree
		tree = MCTS_Tree(start_state, self.size, self.num_actions, root_action_distribution=root_action_distribution, max_depth=self.max_depth, apprentice=self.apprentice, parallel=self.parallel)
		for t in range(self.simulations_per_state):
			if self.apprentice is not None and t % 10 == 0:
				print("running simulation #t = ", t)
			tree.runSingleSimulation()

		return tree.getActionCounts() / self.simulations_per_state

	# Returns a np array of shape [2* batch_size, 6, 5, 5] of input data for white and black
	# and a np array of shape [2* batch_size, num_actions] for white and black
	# Takes in one output for distributions and one for inputs
	def generateExpertBatch(self, outFile1=None, outFile2=None):
		startingStates = []
		# Generate BATCH_SIZE number of random states for each player
		start = time.time()
		starting_inputs = []
		for i in range(self.batch_size):
			startingStates.append(generateRandomState(self.size, 1))
			starting_inputs.append(startingStates[i].channels_from_state())
		for i in range(self.batch_size):
			startingStates.append(generateRandomState(self.size, -1))
			starting_inputs.append(startingStates[i+self.batch_size].channels_from_state())
		end = time.time()
		starting_inputs = np.array(starting_inputs)
		#print ("Time taken to generate", 2 * self.batch_size, "random states: ", (end-start))

		start = time.time()
		# We don't care about P2's action distribution, and the last column of -1's is unnecessary
		if self.parallel:
			dataBatch = self.parallelGenerateDataBatch(startingStates, starting_inputs)
		else:
			dataBatch = self.generateDataBatch(startingStates, starting_inputs)  
		p1Dist = dataBatch[1][:,:-1]
		p2Dist = dataBatch[2][:, :-1]

		input_data = np.zeros((2, self.batch_size, 6, self.size+4, self.size+4))
		# input_data[0] = starting_inputs[0:self.batch_size]
		# input_data[1] = starting_inputs[self.batch_size:]

		distributions = np.zeros((2* self.batch_size, self.num_actions))
		distributions[0:self.batch_size] = p1Dist
		distributions[self.batch_size:] = p2Dist
		end = time.time()
		#print ("Time taken to do MCTS on", 2 * self.batch_size, "states: ", (end-start))


		start = time.time()
		if outFile1 is not None and outFile2 is not None:
			np.save(outFile1, input_data)
			np.save(outFile2, distributions)
		end = time.time()
		#print ("Time to create inputs for apprentice: ", (end-start))
		return (starting_inputs, distributions)


