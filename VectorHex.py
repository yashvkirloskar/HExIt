import numpy as np
import util
from GameUtils import *

class VectorHex:
    def __init__(self, game_size=5):
        self.game_size = game_size

        # Initialize game.vector to be size (6, gs+2, gs+2) as per input specification
        # Layer 0: white pieces, Layer 1: black pieces, Layer 2: black stones connected to north, Layer 3: black stones connected to the south
        # Layer 4: white stones connected to west, Layer 5: white stones connected to east
        # self.board: white is 1, black is 2
        self.vector = np.zeros((6, game_size+2, game_size+2))
        
        self.vector[0, :, 0:2] = 1
        self.vector[0, :, -2:] = 1

        self.vector[1, 0:2, :] = 1
        self.vector[1, -2:, :] = 1
        

        self.vector[2, 0:2, :] = 1

        self.vector[3, -2:, :] = 1

        self.vector[4, 0:2, :] = 1

        self.vector[5, -2:, :] = 1
 
        self.board = np.zeros((game_size, game_size))

        self.turn = 1
        self.winner = None

    def player_move(self, coord):
        if self.winner is not None:
            return None

        x, y = coord
        # Do some checks
        if x >= self.game_size or x < 0 or y >= self.game_size or y < 0:
            return None
        if self.board[x, y] != 0:
            return None

        self.board[x, y] = self.turn

        if self.turn == 1:
            self.vector[0, x+2, y+2] = 1

        elif self.turn == 2:
            self.vector[1, x+2, y+2] = 1

        if check_win(self.board, self.turn, self.game_size) == 'Win':
            print ("Game Over, Player " + str(self.turn) + " wins!")
            self.winner = self.turn
        else:
            self.turn = 3 - self.turn

    def board_from_channels(self):
        # TODO
        return None

    

