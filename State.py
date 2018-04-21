import numpy as np
import VectorHex
from GameUtils import *
from copy import deepcopy, copy
import random

class State:
    def __init__(self, game):
        # Questions: Do we want to store the entire game?
        self.game = VectorHex(board)
        self.channels = self.game.vector.copy()
        self.board = self.game.board.copy()
	
    def __eq__(self, other):
        return np.array_equal(self.board, other)
	
    def __repr__(self):
        return str(self.board)

    def winner(self):
        # TODO
        # May need to change to fit [-1, 1] instead of [1, 2]
        if self.game.winner == 1:
            return 1
        elif self.game.winner == 2:
            return -1
        else:
            return 0

    def isTerminalState(self):
        return self.game.winner is not None

    def calculateReward(self):
        return self.winner()

    def nextState(self, action):
        row, col = action // self.game_size, action % self.game_size
        game = deepcopy(self.game)
        game.player_move((row, col))
        return State(game)

    def isLegalAction(self, action):
        row, col = action // self.game_size, action % self.game_size
        return self.game.isLegal((row, col))

    def legalActions(self):
        la = [a for a in range(self.game_size*self.game_size) if self.isLegalAction(a)]
        return la

    def chooseRandomAction(self):
        if len(self.legalActions()) == 0:
            return -1 # this really shouldn't happen
        return random.choice(self.legalActions())

    def turn(self):
        return self.game.turn

    def isPlayerOneTurn(self):
        return self.game.turn == 1

    def isPlayerTwoTurn(self):
        return self.game.turn == 2

    def nonTerminalActions(self):
        la = self.legalActions()
        nta = []
        for action in la:
            b = self.nextState(action)
            if not b.isTerminalState():
                nta.append(action)
        return nta


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