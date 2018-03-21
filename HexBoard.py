from GameStructures import *

class Board(list):
    # Initialize board and set all the neighbors so that each piece knows who it is next to
    def __init__(self, game_size=5):
        self.game_size=game_size
        self.hexagons = [[Piece(pos=(i,j)) for i in range(self.game_size)] for j in range(self.game_size)]
        self.hexagons[0][0].add_neighbor(self.hexagons[0][1], 2)
        self.hexagons[0][0].add_neighbor(self.hexagons[1][0], 3)
        for i in range(1, self.game_size):
            self.hexagons[i][0].add_neighbor(self.hexagons[i-1][0], 6)
            self.hexagons[0][i].add_neighbor(self.hexagons[0][i-1], 5)
            self.hexagons[i][-1].add_neighbor(self.hexagons[i-1][-1], 6)
            self.hexagons[-1][i].add_neighbor(self.hexagons[-1][i-1], 5)

        for i in range(1, self.game_size-1):
            for j in range(1, self.game_size-1):
                self.hexagons[i][j].add_neighbor(self.hexagons[i-1][j+1], 1)
                self.hexagons[i][j].add_neighbor(self.hexagons[i][j+1], 2)
                self.hexagons[i][j].add_neighbor(self.hexagons[i+1][j], 3)
                self.hexagons[i][j].add_neighbor(self.hexagons[i+1][j-1], 4)
                self.hexagons[i][j].add_neighbor(self.hexagons[i][j-1], 5)
                self.hexagons[i][j].add_neighbor(self.hexagons[i-1][j], 6)

    # Override the list methods to allow for out of range calls
    def __getitem__(self, tup):
        key1, key2 = tup
        if key1 >= self.game_size or key1 < 0 or key2 >= self.game_size or key2 < 0:
            return EmptyPiece()
        else:
            return self.hexagons[key1][key2]

    def __setitem__(self, tup, val):
        key1, key2 = tup
        if key1 >= self.game_size or key1 < 0 or key2 >= self.game_size or key2 < 0:
            pass
        else:
            self.hexagons[key1][key2].player = val

    def __delitem__(self, tup):
        key1, key2 = tup
        if key1 >= self.game_size or key1 < 0 or key2 >= self.game_size or key2 < 0:
            pass
        else:
            self.hexagons[key1][key2].player = -1

    def __str__(self):
        board = [[0 for i in range(self.game_size)] for j in range(self.game_size)]
        for i in range(self.game_size):
            for j in range(self.game_size):
                if self.hexagons[i][j].player != -1:
                    board[i][j] = self.hexagons[i][j].player+1
        out = ""
        for i in range(self.game_size):
            out += str(board[i])+"\n"
        return out

