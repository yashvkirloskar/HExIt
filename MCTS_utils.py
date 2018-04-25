import util
import numpy as np
from State import State, TestState
from MCTS_Tree import MCTS_Tree, MCTS_Node
import random, math
from scipy.special import comb
from copy import deepcopy

# get_config_proportions(i) is the number of possible board configurations
# given that num_squares_per = i
def get_config_proportions(size, turn):
    counts = np.zeros(size)
    for i in range(size):
        # Number of board configurations if no moves made 
        if i == 0:
            counts[i] = size
        # Number of board configurations for other choices given P1 turn
        elif i < size-1:
            counts[i] = size * (size+i)
        # If only one move available, should not add count for P1, since P1 playing will be a terminal state
        else:
            if turn == 1:
                counts[i] = size**2
            # Ensures non-terminal state
            else:
                counts[i] = 0
        counts[i] *= (size-i)
    # Normalize to represent probability distribution
    return counts/np.sum(counts)

# Must return a randomly generated non-terminal state
def generateRandomState(size, turn):
    # turn = 1 if P1's turn, -1 if P2's turn
    # Number of pieces both players have on the board // 2
    num_squares_per = np.random.choice((size**2) // 2, p=get_config_proportions(size**2//2, turn))
    # Turn offset should be 0 if P1 turn, 1 if P2 turn
    turn_offset = (turn - 1) // -2
    board = State(np.zeros((size, size)))
    for i in range(num_squares_per*2+turn_offset):
        action = board.chooseRandomAction()
        next = board.nextState(action)
        if i >= size*2:
            if next.isTerminalState():
                if board.turn() == turn:
                    return board
                else:
                    return prev
        prev = board
        board = next

    return board
    
def board_from_channels(channels):
    whites = channels[0]
    blacks = channels[1] + np.ones_like(channels[1])*(channels[1] == 1)
    board = whites+blacks
    return board[2:-2, 2:-2]