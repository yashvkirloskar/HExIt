import util
import numpy as np
from State import State

def action_string(action):
	if action == 9:
		return "dummy"
	row, col = action // 3, action % 3
	row_string = "top" if row == 0 else ("middle" if row == 1 else "bottom")
	col_string = "left" if col == 0 else ("center" if col == 1 else "right")
	s = row_string + ", " + col_string
	return s


class MCTS_Tree:
	def __init__(self, start_state, size, num_actions, root_action_distribution, max_depth=6, apprentice=None):
		self.start_state = start_state
		self.size = size
		self.action_counts = np.zeros(num_actions)
		self.root = MCTS_Node(start_state, size, num_actions, max_depth=max_depth, apprentice=apprentice, isRoot=True, root_action_distribution=root_action_distribution)

		self.apprentice = apprentice
		self.root_action_distribution = root_action_distribution

	# Runs a single simulation starting from the root
	# Update the action_counts array
	def runSingleSimulation(self):
		action_from_root, reward = self.root.runSimulation()
		if action_from_root != -1:
			self.action_counts[action_from_root] += 1

	# Returns the action counts, which is a list.
	# The i-th element is the number of time action i was chosen from the root.
	def getActionCounts(self):
		return self.action_counts



class MCTS_Node:
	def __init__(self, state, size, num_actions, parent=None, max_depth=6, apprentice=None, isRoot=False, root_action_distribution=None):
		self.state = state
		self.size = size
		self.num_actions = num_actions
		self.parent = parent
		self.max_depth = max_depth
		self.apprentice = apprentice

		self.children = [None for i in range(num_actions)]
		self.node_visits = 0 # N(S)
		self.outgoing_edge_traversals = np.zeros(num_actions) # N(S, A)
		self.outgoing_edge_rewards = np.zeros(num_actions) # R(S, A)

		self.apprentice_probs = np.zeros(num_actions)
		self.calculated_apprentice_probs = False # Helps avoid redundant queries to the apprentice

		self.c_b = 3 # Needs tuning, coefficient for the term in UCT meant to incentivize exploration of never before seen moves
		self.w_a = 40 # Needs tuning, coefficient for influence of apprentice in UCT (should be roughly avg # simulations per action at root)
		self.epsilon = 1e-8 # To prevent division by 0 in UCT calculation

		self.isRoot = isRoot
		self.root_action_distribution = root_action_distribution

	def __repr__(self):
		s = self.state.__repr__()# + "R(S, A): " + str(self.outgoing_edge_rewards) + "\nN(S, A): " + str(self.outgoing_edge_traversals) + "\nN(S): " + str(self.node_visits)
		return s
 
	# Runs one simulation from this node till the end of the game.
	# Performs the reward propagation from the end up till this node.
	# Recall that Player 1 wants to maximize reward (and thus UCT), while Player 2 seeks to minimize it.
	# Returns an action (an integer from 0 to NUM_ACTIONS -1).
	# The action is the optimal move for the player whose turn it is at this state.
	# This action represents the action chosen at this node.
	def runSimulation(self, depth=0):
		
		# Check if this is a terminal state, and if so calculate and return reward
		if self.state.isTerminalState():
			# Reward is positive for a White win, and negative for a Black win
			reward = self.state.calculateReward()
			self.updateStatistics(-1, reward) # No-op since I am a leaf node.
			return (-1, reward) # -1 is just to signify that the chosen action is irrelevant. Could be any number.

		# rollout if at max depth
		if depth >= self.max_depth:
			random_action, reward = self.rollout() # Rollout should perform stats update
			return (random_action, reward)

		# choose best action, according to UCT. chooseBestAction should account for Black or White.
		chosen_action = self.chooseBestAction() # Guaranteed to be a legal action
		# Grab the resulting next state
		next_state_node = self.children[chosen_action]

		# If we've never been to this next state node, create it and begin rollout.
		if next_state_node is None:
			next_state = self.state.nextState(chosen_action) # find what state results
			new_node = MCTS_Node(next_state, self.num_actions, parent=self, max_depth=self.max_depth, apprentice=self.apprentice) # create a node for it
			self.children[chosen_action] = new_node # set this node as one of my children
			_, reward = new_node.rollout() # Rollout will take care of stats updates for the new child node.
			self.updateStatistics(chosen_action, reward) # Update my own stats

			return (chosen_action, reward)
		# If we have been to this next state node, recurse
		else:
			_, reward = next_state_node.runSimulation(depth=depth+1)
			self.updateStatistics(chosen_action, reward)
			return (chosen_action, reward)


	# Given an action A and a reward R, appropriately updates N(S, A) and R(S, A)
	# If I am a leaf node (terminal state, no need to do anything)
	def updateStatistics(self, action, reward):
		self.node_visits += 1
		if self.state.isTerminalState():
			return
		if action != -1:
			self.outgoing_edge_traversals[action] += 1
			self.outgoing_edge_rewards[action] += reward


	def rollout(self):
		# If this state is terminal, calculate and return the reward, and return the dummy action.
		if self.state.isTerminalState():
			reward = self.state.calculateReward()
			self.updateStatistics(-1, reward)
			return (-1, reward)

		random_action = self.state.chooseRandomAction() # Guaranteed to be a legal action
		# Grab the node for the resulting state, creating one if necessary
		next_state_node = self.children[random_action]
		if next_state_node is None:
			next_state = self.state.nextState(random_action)
			next_state_node = MCTS_Node(next_state, self.num_actions, parent=self, max_depth=self.max_depth, apprentice=self.apprentice)
			self.children[random_action] = next_state_node

		# Recursively call rollout on that next state node.
		_, reward = next_state_node.rollout()
		# update my stats
		self.updateStatistics(random_action, reward)
		return (random_action, reward)


	def chooseBestAction(self):
		uct = np.zeros(self.num_actions)
		if self.node_visits != 0:
			uct = self.computeUct()

		apprentice_term = np.zeros(self.num_actions)
		if self.apprentice is not None:
			if self.isRoot:
				numer = self.root_action_distribution
			else:
				stateInput = np.ones((2, 6, self.size, self.size))
				stateInput[(self.state.turn-1)*-1] = self.state.channels_from_state()
				numer = self.apprentice.getActionDistribution(stateInput) # vector
			denom = self.outgoing_edge_traversals + 1 # vector
			apprentice_term = self.w_a * (numer[(self.state.turn-1)*-1]/denom)

		uct_new = uct + apprentice_term

		# generate mask (0 if illegal, 1 if legal)
		legal_actions = self.state.legalActions()
		mask = np.array([(1 if action in legal_actions else 0) for action in range(25)])
		legal_ucts = uct_new * mask

		return np.argmax(legal_ucts)


	def computeUct(self):
		avg_reward = self.outgoing_edge_rewards / (self.outgoing_edge_traversals + self.epsilon) # vector
		if (self.state.isPlayerTwoTurn()):
			avg_reward *= -1
		numer = np.log(self.node_visits) # scalar
		denom = self.outgoing_edge_traversals + self.epsilon # vector
		new_action_incentive = self.c_b * np.sqrt(numer / denom) # vector
		uct = avg_reward + new_action_incentive # vector
		return uct

