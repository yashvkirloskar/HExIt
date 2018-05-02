import util
import numpy as np
from MCTS_Tree import MCTS_Tree, MCTS_Node
from State import State
from MCTS_utils import *
import time
import multiprocessing, logging
from multiprocessing import Pool
import logging
import threading
import sys

class MCTS:
    #uct_new(s,a) = uct(s,a) + w_a * (apprentice_policy(s,a) / n(s,a) + 1)
    # given a list of batch_size state objects
    # return an array with batch_size elements.  Each element is a 26-list.  The state, followed by number of times we took action i
    #uct(s,a) = r(s,a)/n(s,a) + c_b * sqrt ( log(n(s)) / n(s,a) )

    def __init__(self, size=5, batch_size=256, simulations_per_state=1000, max_depth=6,
        apprentice=None, parallel=False, threaded=False, num_threads=16):
        self.size = size
        self.num_actions = size**2
        self.batch_size = batch_size
        self.simulations_per_state = simulations_per_state
        self.max_depth = max_depth
        self.apprentice = apprentice
        self.parallel = parallel
        self.threaded = threaded
        self.num_threads = batch_size

    # Runs SIMULATIONS_PER_STATE simulations on the given state, 
    # collects the action distribution, and returns the argmax action.
    def getMove(self, state):
        # run all the starting states through the apprentice once, for efficiency
        temp = self.threaded
        self.threaded = False
        root_action_distribution = [None for i in range(self.num_actions)]
        if self.apprentice is not None:
            state_input = state.channels_from_state()
            root_action_distribution = self.apprentice.getActionDistributionSingle(state_input, state.turn())
            # this is a [num_actions,] shaped vector

        action_distribution = self.runSimulations(state, root_action_distribution)
        self.threaded = temp
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
            # print("root_action_distributions shape:", root_action_distributions.shape)
            # this is a [batch_size, num_actions] shaped matrix

        for i, state in enumerate(starting_states):
            # print("i:", i)
            if state.isPlayerOneTurn():
                action_distribution1[i][0:self.num_actions] = self.runSimulations(state, root_action_distributions[0][i])
                action_distribution2[i][-1] = 1
            else:
                action_distribution2[i-self.batch_size][0:self.num_actions] = self.runSimulations(state, root_action_distributions[1][i - self.batch_size])
                action_distribution1[i-self.batch_size][-1] = 1

        return (starting_states, action_distribution1, action_distribution2)

    def master_thread_func(self):
        if self.apprentice is None:
            return 

        #while (self.num_white_threads_left + self.num_black_threads_left) > 0:
        while self.num_live_threads > 0:
            self.num_submitted = 0
            self.apprentice_batch = np.zeros((2 * self.num_threads, 6, self.size + 4, self.size + 4))
            self.apprentice_predictions = np.zeros((2, self.num_threads, self.num_actions))
            # Wait for the batch_ready event
            self.batch_ready.wait()

            if self.finished:
                return

            # ship the apprentice off
            self.apprentice_finished.clear()
            self.apprentice_predictions = self.apprentice.getActionDistribution(self.apprentice_batch)

            self.batch_ready.clear()
            self.apprentice_finished.set()

        return


    # Suppose there are T threads (pre-determined) there are N threads to process
    # Then this one will take care of every state such that N % T = thread_num
    def worker_thread_func(self, thread_num):
        for i, state in enumerate(self.starting_states):
            if i % self.num_threads != thread_num:
                continue
            if state.isPlayerOneTurn():
                self.action_distribution1[i][0:self.num_actions]= self.runSimulations(
                    state, self.root_action_distributions[0][i],
                    batch_num=i % self.batch_size)
                self.action_distribution2[i][-1] = 1

            else:
                self.action_distribution2[i-self.batch_size][0:self.num_actions] = self.runSimulations(
                    state, self.root_action_distributions[1][i - self.batch_size],
                    batch_num=i % self.batch_size)
                self.action_distribution1[i-self.batch_size][-1] = 1

            # logging.debug("Finished processing state # " + str(i))
            

        # logging.debug("Worker Thread returning with " +
        #     str(self.num_live_threads) + " threads live." +
        #     " Num submitted is " + str(self.num_submitted))

        # Decrement num_live_threads
        self.num_live_threads -= 1

        if self.num_submitted == self.num_live_threads:
            self.batch_ready.set()

        return

    def threadedGenerateDataBatch(self, starting_states, starting_inputs):
        self.starting_states = starting_states
        self.starting_inputs = starting_inputs

        # Initialize batch lock, batch_ready event, apprentice_finished event
        self.batch_lock = threading.Lock()
        self.batch_ready = threading.Event()
        self.apprentice_finished = threading.Event()
        self.num_submitted = 0
        self.finished = False

        # Configure the log
        # logging.basicConfig(level=logging.DEBUG,
        #             format='(%(threadName)-10s) %(message)s',
        #             )

        # Initialize the threshold which is the number of states the master thread should wait for 
        # before shipping the batch off to the apprentice
        # This will get decremented when certain threads finish all their simulations
        self.num_live_threads = self.num_threads
        # Initialize the apprentice batch that will shipped off to the apprentice for prediction
        #self.apprentice_batch = np.zeros((2 * self.batch_size, 6, self.size + 4, self.size + 4))
        self.apprentice_batch = np.zeros((2 * self.num_threads, 6, self.size + 4, self.size + 4))

        # Initialize the matrix of action distributions that the apprentice will periodically fill
        #self.apprentice_predictions = np.zeros((2 * self.batch_size, self.num_actions))
        self.apprentice_predictions = np.zeros((2, self.num_threads, self.num_actions))

        action_distribution1 = np.zeros(shape=(self.batch_size, self.num_actions + 1))
        action_distribution2 = np.zeros(shape=(self.batch_size, self.num_actions + 1))
        self.action_distribution1 = action_distribution1
        self.action_distribution2 = action_distribution2

        # run all the starting states through the apprentice as once
        #root_action_distributions = [None for i in range(self.num_actions)]
        root_action_distributions = np.zeros((2, self.batch_size, self.num_actions))
        if self.apprentice is not None:
            root_action_distributions = self.apprentice.getActionDistribution(starting_inputs)
            # this is a [batch_size, num_actions] shaped matrix
        self.root_action_distributions = root_action_distributions

        # Spawn the master thread
        master_thread = threading.Thread(name='Master Thread', 
                      target=self.master_thread_func,
                      args=())
        # Start the master thread
        master_thread.start()

        # Spawn some worker threads
        worker_threads = []
        for i in range(self.num_threads):
            worker_thread = threading.Thread(name='Worker Thread ' + str(i), 
                          target=self.worker_thread_func,
                          args=(i,))
            worker_threads.append(worker_thread)
            worker_thread.start()


        # Wait for all the threads to finish
        for i in range(self.num_threads):
            worker_threads[i].join()

        # Once all worker threads return, set the batch_ready event so the master can return
        self.finished = True
        self.batch_ready.set()
        master_thread.join()


        return (starting_states, action_distribution1, action_distribution2)

    # Runs SIMULATIONS_PER_STATE simulations, each starting from the given START_STATE.
    # Returns a list with as many elements as there are actions, plus 1. (ex. 26 for a 5x5 hex game).
    # Each element is a probability (between 0 and 1).
    # The i-th element is the number of times we took the i-th action from the root state (as a probability).
    # The last element is the number of times we took no action (if it wasn't this player's turn.)
    def runSimulations(self, start_state, root_action_distribution, batch_num=0):
        # Initialize new tree
        tree = MCTS_Tree(start_state, self.size, self.num_actions,
            root_action_distribution=root_action_distribution,
            max_depth=self.max_depth, apprentice=self.apprentice, parallel=self.parallel, threaded=self.threaded, batch_num=batch_num, parent=self)
        
        for t in range(self.simulations_per_state):
            # print (t)
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

        start = time.time()
        # We don't care about P2's action distribution, and the last column of -1's is unnecessary
        if self.parallel:
            dataBatch = self.parallelGenerateDataBatch(startingStates, starting_inputs)
        elif self.threaded:
            dataBatch = self.threadedGenerateDataBatch(startingStates, starting_inputs)
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

        start = time.time()
        if outFile1 is not None and outFile2 is not None:
            np.save(outFile1, input_data)
            np.save(outFile2, distributions)
        end = time.time()
        return (starting_inputs, distributions)
