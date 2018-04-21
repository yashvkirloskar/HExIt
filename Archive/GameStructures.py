class EmptyPiece:
    def __init__(self):
        self.player = -1

    def get_neighbors():
        return []


# Data structure to hold information about pieces
# Useful for doing bfs
class Piece:
    def __init__(self, player=-1, pos=None, neighbor1=None, neighbor2=None, neighbor3=None, neighbor4=None, neighbor5=None, neighbor6=None):
        assert(player in [-1, 0, 1])
        self.player = player
        self.pos = pos
        self.neighbor1 = None
        self.neighbor2 = None
        self.neighbor3 = None
        self.neighbor4 = None
        self.neighbor5 = None
        self.neighbor6 = None

    # Add neighbor to piece, and add the reverse too
    def add_neighbor(self, piece, neighborNo):
        if neighborNo == 1:
            self.neighbor1 = piece
            piece.neighbor4 = self
        elif neighborNo == 2:
            self.neighbor2 = piece
            piece.neighbor5 = self
        elif neighborNo == 3:
            self.neighbor3 = piece
            piece.neighbor6 = self
        elif neighborNo == 4:
            self.neighbor4 = piece
            piece.neighbor1 = self
        elif neighborNo == 5:
            self.neighbor5 = piece
            piece.neighbor2 = self
        elif neighborNo == 6:
            self.neighbor6 = piece
            piece.neighbor3 = self

    # Return only the neighbors of the piece that are of the same team, and that belong to a playing player's team
    def get_neighbors(self):
        neighbors = [self.neighbor1, self.neighbor2, self.neighbor3, self.neighbor4, self.neighbor5, self.neighbor6]
        return [i for i in neighbors if i is not None and i.player == self.player and i.player != -1]

# # Board Util to deal with overflow
# class Board(list):
#     def __init__(self, size=5):
#         self.size = size
#         self.grid = [[EmptyPiece() for i in range(self.size)] for j in range(self.size)]

#     def __getitem__(self, tup):
#         key1, key2 = tup
#         if key1 >= self.size or key1 < 0 or key2 >= self.size or key2 < 0:
#             return EmptyPiece()
#         else:
#             return self.grid[key1][key2]

#     def __setitem__(self, tup, val):
#         key1, key2 = tup
#         if key1 >= self.size or key1 < 0 or key2 >= self.size or key2 < 0:
#             pass
#         else:
#             self.grid[key1][key2] = val

#     def __delitem__(self, key):
#         key1, key2 = tup
#         if key1 >= self.size or key1 < 0 or key2 >= self.size or key2 < 0:
#             pass
#         else:
#             self.grid[key1][key2] = EmptyPiece()