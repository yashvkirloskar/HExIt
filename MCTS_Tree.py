import util
import numpy as np
from State import State

class MCTS_Tree:
	def __init__(self, start_state, num_actions, max_depth=6, apprentice=None):
		self.start_state = start_state
		self.action_counts = np.zeros(num_actions)
		self.root = MCTS_Node(start_state, num_actions, max_depth=max_depth, apprentice=apprentice)
		self.apprentice = apprentice

	# Runs a single simulation starting from the root
	# Update the action_counts array
	def runSingleSimulation(self):
		reward, action_from_root = self.root.runSimulation()
		self.action_counts[action_from_root] += 1

	# Returns the action counts, which is a list.
	# The i-th element is the number of time action i was chosen from the root.
	def getActionCounts(self):
		return self.action_counts

#uct(s,a) = r(s,a)/n(s,a) + c_b * sqrt ( log(n(s)) / n(s,a) )
	#uct_new(s,a) = uct(s,a) + w_a * (apprentice_policy(s,a) / n(s,a) + 1)



# If a state is White's move, Black should choose NO_ACTION every time
# uct should 
class MCTS_Node:
	def __init__(self, state, num_actions, parent=None, max_depth=6, apprentice=None):
		self.state = state
		self.num_actions = num_actions
		self.parent = parent
		self.max_depth = max_depth
		self.apprentice = apprentice

		self.children = [None for i in range(num_actions)]
		self.node_visits = 0 # N(S)
		self.outgoing_edge_traversals = np.zeros(num_actions) # N(S, A)
		self.outgoing_edge_rewards = np.zeros(num_actions) # R(S, A)

		self.have_visited_before = False # This could help speed up computations for never-before-seen nodes

	# Runs one simulation from this node till the end of the game.
	# Performs the reward propagation from the end up till this node.
	# Returns an action (an integer from 0 to NUM_ACTIONS -1).
	# This action represents the action chosen at this node.
	def runSimulation(self, depth=0):
		
		self.node_visits += 1

		# Check if this is a terminal state
		if self.state.isTerminalState():
			reward = self.state.calculateReward() # The reward function should calculate the reward for the player that just played this move
			return (reward, -1) # -1 is just to signify that the chosen action is irrelevant. Could be any number

		# rollout
		if depth >= self.max_depth:
			reward, random_action = self.rollout()
			self.updateStatistics(random_action, reward)
			return (reward, random_action)

		# choose best action
		# if we don't know what to choose, rollout randomly till end of simulation
		chosen_action = self.chooseBestAction()
		if chosen_action == -1:
			reward, random_action = self.rollout()
			self.updateStatistics(random_action, reward)
			return (reward, random_action)
		
		# recursively continue the simulation from the next node
		next_state_node = self.children[chosen_action]
		if next_state_node == None:
			next_state = self.state.nextState(chosen_action)
			next_state_node = MCTS_Node(next_state, num_actions, parent=self, max_depth=max_depth, apprentice= apprentice)
			self.children[chosen_action] = next_state_node

		# grab the reward, update reward and visit counts
		reward, _ = next_state_node.runSimulation(depth=depth + 1)
		self.updateStatistics(chosen_action, reward)

		return (reward, chosen_action)


	def updateStatistics(self, action, reward):
		self.outgoing_edge_traversals[action] += 1
		self.outgoing_edge_rewards[action] += reward

	def rollout(self):
		if self.state.isTerminalState():
			reward = 
		action, next_state = self.state.chooseRandomAction()
		next_state_node = MCTS_Node(next_state, num_actions, parent=self, max_depth=max_depth, apprentice=apprentice)
		self.children[action] = next_state_node
		reward, _ = next_state_node.rollout()
		# update stats
		return (reward, action)

	def chooseBestAction(self):
