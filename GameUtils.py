# Utility functions for checking game conditions
import util
import numpy as np

def isLegal(board, game_size, coord):
    x, y = coord
    if x >= game_size or x < 0 or y >= game_size or y < 0:
        return False
    if board[x, y] != 0:
        return False
    return True

# Do a BFS on the just played player to see if they have met the win condition
def isWin(board, turn, game_size):
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

def get_neighbors(board, coord, turn):
    neighbors = []
    i, j = coord
    indices = hex_indices(i+1, j+1)

    temp = np.pad(board.copy(), 1, 'constant', constant_values=-1)
    for index in indices:
        if temp[index] == -1 or temp[index] != turn:
            continue
        else:
            neighbors.append((index[0]-1, index[1]-1))
    return neighbors

def hex_indices(i, j):
    indices = []
    indices.append((i-1, j+1))
    indices.append((i, j+1))
    indices.append((i+1, j))
    indices.append((i+1, j-1))
    indices.append((i, j-1))
    indices.append((i-1, j))
    return indices

def board_from_channels(channels):
    whites = channels[0]
    blacks = channels[1] + np.ones_like(channels[1])*(channels[1] == 1)
    board = whites+blacks
    return board[2:-2, 2:-2]