# Logic and checks for Hex game
# Should handle moves, and win conditions
# No UI built in yet
import util

class Hex: 
    def __init__(self, game_size=5, player0='user', player1='user'):
        # Initialize game variables
        self.game_size = 5
        self.board = Board(self.game_size)
        self.player0 = player0
        self.player1 = player1
        self.turn = 0
        self.winner = None
        assert(self.player0 in ['user', 'ai'] and self.player1 in ['user', 'ai'])

    

    def player_move(self, coord):
        if self.winner is not None:
            return None
        x, y = coord

        # Check that board is not being overwritten
        assert(isinstance(self.board[x, y], EmptyPiece))
        if self.turn == 0:
            self.board[x, y] = Piece(self.turn, (x,y), self.board[x+1, y], self.board[x, y+1], self.board[x-1, y], self.board[x, y-1])
        elif self.turn == 1:
            self.board[x, y] = Piece(self.turn, (x,y), self.board[x+1, y], self.board[x, y+1], self.board[x-1, y], self.board[x, y-1])

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
                        if not isinstance(child, EmptyPiece):
                            fifo.push(child)

class EmptyPiece:
    def __init__(self):
        self.player = -1

    def get_neighbors():
        return []


# Data structure to hold information about pieces
# Useful for doing bfs
class Piece:
    def __init__(self, player=None, pos=None, neighbor1=None, neighbor2=None, neighbor3=None, neighbor4=None):
        assert(player in [0, 1])
        self.player = player
        self.pos = pos
        if neighbor1.player == self.player:
            self.neighbor1 = neighbor1
            self.neighbor1.neighbor3 = self
        else:
            self.neighbor1 = EmptyPiece()

        if neighbor2.player == self.player:
            self.neighbor2 = neighbor2
            self.neighbor2.neighbor4 = self
        else:
            self.neighbor2 = EmptyPiece()

        if neighbor3.player == self.player:
            self.neighbor3 = neighbor3
            self.neighbor3.neighbor1 = self
        else:
            self.neighbor3 = EmptyPiece()

        if neighbor4.player == self.player:
            self.neighbor4 = neighbor4
            self.neighbor4.neighbor2 = self
        else:
            self.neighbor4 = EmptyPiece()

    def get_neighbors(self):
        return [self.neighbor1, self.neighbor2, self.neighbor3, self.neighbor4]

# Board Util to deal with overflow
class Board(list):
    def __init__(self, size):
        self.size = size
        self.grid = [[EmptyPiece() for i in range(self.size)] for j in range(self.size)]

    def __getitem__(self, tup):
        key1, key2 = tup
        if key1 >= self.size or key1 < 0 or key2 >= self.size or key2 < 0:
            return EmptyPiece()
        else:
            return self.grid[key1][key2]

    def __setitem__(self, tup, val):
        key1, key2 = tup
        if key1 >= self.size or key1 < 0 or key2 >= self.size or key2 < 0:
            pass
        else:
            self.grid[key1][key2] = val

    def __delitem__(self, key):
        key1, key2 = tup
        if key1 >= self.size or key1 < 0 or key2 >= self.size or key2 < 0:
            pass
        else:
            self.grid[key1][key2] = EmptyPiece()

