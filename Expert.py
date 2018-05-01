import numpy as np 
from MCTS import MCTS
import multiprocessing, logging
from multiprocessing import Pool
from copy import deepcopy

class Expert:

    def __init__(self, board_size=5, batch_size=256, simulations_per_state=1000, max_depth=6, apprentice=None):
        self.board_size = board_size
        self.batch_size = batch_size
        self.simulations_per_state = simulations_per_state
        self.max_depth = max_depth
        self.apprentice = apprentice

        self.mcts = MCTS(size=board_size, batch_size=batch_size, simulations_per_state=simulations_per_state, max_depth=max_depth, apprentice=apprentice, parallel=False, threaded=True, num_threads=16)


    def generateBatch(self):
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(logging.INFO)
        p = Pool()
        cpy = deepcopy(self.mcts)
        ret = p.apply(cpy.generateExpertBatch, [None, None])
        p.close()
        p.join()
        return ret


    def getMove(self, state):
        print ("Getting move from expert")
        return self.mcts.getMove(state)
        
