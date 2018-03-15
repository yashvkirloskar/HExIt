# Logic and checks for Hex game
# Should handle moves, and win conditions
# No UI built in yet
import util
from GameStructures import *
from HexBoard import *

class Hex: 
    def __init__(self, game_size=5, player0='user', player1='user'):
        # Initialize game variables
        self.game_size = game_size
        self.board = Board(game_size)
        self.player0 = player0
        self.player1 = player1
        self.turn = 0
        self.winner = None
        assert(self.player0 in ['user', 'ai'] and self.player1 in ['user', 'ai'])

    # Make a move
    def player_move(self, coord):
        if self.winner is not None:
            return None
        x, y = coord

        # Check that board is not being overwritten
        assert(self.board[x, y].player == -1)
        self.board[x, y].player = self.turn

        if self.check_win() == 'Win':
            print ("Game Over, Player " + str(self.turn) + " wins!")
            self.winner = self.turn
        else:
            self.turn = 1 - self.turn


    # Do a BFS on the just played player to see if they have met the win condition
    def check_win(self):
        fifo = util.Queue()
        closed = util.Counter()
        
        for i in range(self.game_size):
            if self.turn == 0:
                if self.board[0, i].player == self.turn:
                    fifo.push(self.board[0, i])
            elif self.turn == 1:
                if self.board[i, 0].player == self.turn:
                    fifo.push(self.board[i, 0])

        while True:
                if fifo.isEmpty():
                    return "Not a Win"
                node = fifo.pop()
                if self.turn == 0:
                    if node.pos[0] == self.game_size-1:
                        return "Win"
                elif self.turn == 1:
                    if node.pos[1] == self.game_size-1:
                        return "Win"
                if closed[node] == 0:
                    closed[node] += 1
                    for child in node.get_neighbors():
                        fifo.push(child)


