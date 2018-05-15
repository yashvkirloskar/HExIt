import numpy as np
import util
from GameUtils import *
from HexAgent import *
from State import *

class VectorHex:
    def __init__(self, game_size=5, channels=None, p1='ai', p2='ai'):
        self.game_size = game_size
        self.board = np.zeros((game_size, game_size))
        self.turn = 1
        self.winner = None
        self.p1 = p1
        self.p2 = p2
        self.ai = HexAgent("bestAgent", game_size, 256, 10, 6)

    def player_move(self, coord):
        if self.winner is not None:
            return None

        x, y = coord
        # Do some checks
        if not isLegal(self.board, self.game_size, coord):
            return None

        self.board[x, y] = self.turn

        if self.isWin(self.board, self.turn, self.game_size):
            print ("Game Over, Player " + str(self.turn) + " wins!")
            self.winner = self.turn
        else:
            self.turn = 3 - self.turn

    def ai_move(self):
        if self.winner is not None:
            return None
        state_board = self.board.copy()
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[0]):
                if state_board[i, j] == 2:
                    state_board[i, j] = -1
        if self.p1 == 'random' and self.turn == 1:
            ai_move = self.ai.getRandomMove((State(state_board)))
        elif self.p2 == 'random' and self.turn == 2:
            ai_move = self.ai.getRandomMove((State(state_board)))
        else:
            ai_move = self.ai.getMove(State(state_board))
            
        
        self.board[int(float(ai_move) // self.board.shape[0]), int(float(ai_move) % self.board.shape[0])] = self.turn
        print (self.board)
        print ("AI move: ", ai_move)
        if self.isWin(self.board, self.turn, self.game_size):
            print ("Game Over, Player " + str(self.turn) + " wins!")
            self.winner = self.turn
        self.turn = 3-self.turn

        return (3-self.turn, ai_move)


    def isWin(self, board, turn, game_size):
        fifo = util.Queue()
        closed = util.Counter()
        
        for i in range(game_size):
            # White pieces on left edge
            if turn == 1:
                if board[i, 0] == turn:
                    fifo.push((i, 0))
            # Black pieces on North edge
            elif turn == 2:
                if board[0, i] == turn:
                    fifo.push((0, i))

        while True:
                if fifo.isEmpty():
                    return False
                node = fifo.pop()
                if turn == 1:
                    if node[1] == game_size-1:
                        return True
                elif turn == 2:
                    if node[0] == game_size-1:
                        return True
                if closed[node] == 0:
                    closed[node] += 1
                    for child in get_neighbors(board, node, turn):
                        fifo.push(child)

        

