import util
import numpy as np
from State import State, TestState
from MCTS import MCTS
from MCTS_Tree import MCTS_Tree, MCTS_Node
import random, math
from scipy.special import comb

# get_config_proportions(i) is the number of possible board configurations
# given that num_squares_per = i
def get_config_proportions(size):
    counts = np.zeros(size)
    for i in range(size):
        # Number of board configurations if no moves made, or if P1 has only made a move
        if i == 0:
            counts[0] = size**2+1
        # Number of board configurations for other choices
        elif i < size**2 // 2:
            counts[i] = (comb(size**2, i*2) * comb(i*2, i)) + (comb(size**2, i*2 + 1) * comb(i*2 + 1, i))
        # If only one move available, should not add count for P1, since P1 playing will be a terminal state
        else:
            counts[i] = comb(size**2, i*2) * comb(i*2, i)
    # Normalize to represent probability distribution
    return counts/np.sum(counts)

# Must return a randomly generated non-terminal state
def generateRandomState(size):
    # turn = 1 if P1's turn, -1 if P2's turn
    turn = random.choice([1,-1])
    # Number of pieces both players have on the board // 2
    num_squares_per = np.random.choice((size**2) // 2, p=get_config_proportions(size))
    
    # Ensure non-terminal state
    if num_squares_per == size**2 // 2:
        turn = 1

    board = State(np.zeros((size, size)))
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