import numpy as np
import VectorHex
from GameUtils import *
from copy import deepcopy, copy
import random
from State_Utils import *

class State:
    def __init__(self, board):
        self.board = board.astype(float)
        # self.curr_turn 1 if P1 turn, and -1 if P2 turn
        self.curr_turn = 1 if np.sum(board) == 0 else -1
        if np.count_nonzero(self.board) < board.shape[0]*2-1:
            self.win = 0
        else:
            self.win = None
	
    def __eq__(self, other):
        return np.array_equal(self.board, other.board)
	
    def __repr__(self):
        return str(self.board)

    def winner(self):
        if isWin(self.board, 1, self.board.shape[0]):
            self.win = 1
            return 1
        if isWin(self.board, -1, self.board.shape[0]):
            self.win = -1
            return -1
        
        self.win = 0

        return 0

    def isTerminalState(self):
        if self.win is None:
            term = self.winner()
            return term != 0
        else:
            return self.win != 0

    def calculateReward(self):
        if self.win is None:
            return self.winner()
        else:
            return self.win


    def nextState(self, action):
        # Assume action is legal
        row, col = action // self.board.shape[0], action % self.board.shape[0]
        cpy = self.board.copy()
        cpy[row, col] = self.curr_turn
        return State(cpy)

    def isLegalAction(self, action):
        row, col = action // self.board.shape[0], action % self.board.shape[0]
        return self.board[row, col] == 0

    def legalActions(self):
        if self.win != 0 and self.winner() != 0:
            return []
        la = np.where(self.board.flatten() == 0)[0]
        return la

    def chooseRandomAction(self):
        la = self.legalActions()
        if len(la) == 0:
            return -1 # this could happen if terminal state
        return random.choice(la)

    def turn(self):
        return self.curr_turn

    def isPlayerOneTurn(self):
        return self.curr_turn == 1

    def isPlayerTwoTurn(self):
        return self.curr_turn == -1

    def nonTerminalActions(self):
        la = self.legalActions()
        nta = []
        for action in la:
            b = self.nextState(action)
            if not b.isTerminalState():
                nta.append(action)
        return nta

    def channels_from_state(self):
        output = np.zeros(6, self.board.shape[0]+4, self.board.shape[0]+4)
        # First layer: All locations of white stones (P1 locations)
        output[0, :, 0:2] = 1
        output[0, :, -2:] = 1
        output[0, 2:-2, 2:-2] = (self.board==1)*self.board
    
        # Second layer: All locations of black stones (P2 locations)
        output[1, 0:2, :] = 1
        output[1, -2:, :] = 1
        output[1, 2:-2, 2:-2] = (self.board==-1)*self.board*-1

        # Third layer: White Stones connected to West Edge (left side)
        output[2, 0:2, :] = 1
        output[2, 2:-2, 2:-2] = self.board * bfs_right(self.board, self.board.size[0])

        output[3, -2:, :] = 1
        output[4, 0:2, :] = 1
        output[5, -2:, :] = 1

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