# Utility functions for checking game conditions
import util
import numpy as np

def hex_indices(i, j):
    indices = []
    indices.append((i-1, j+1))
    indices.append((i, j+1))
    indices.append((i+1, j))
    indices.append((i+1, j-1))
    indices.append((i, j-1))
    indices.append((i-1, j))
    return indices

def get_neighbors(board, coord, turn):
    neighbors = []
    i, j = coord
    indices = hex_indices(i+1, j+1)

    temp = np.pad(board.copy(), 1, 'constant', constant_values=-2)
    for index in indices:
        if temp[index] == -2 or temp[index] != turn:
            continue
        else:
            neighbors.append((index[0]-1, index[1]-1))
    return neighbors

# BFS for white pieces connected to west edge
def bfs_right(board, game_size):
    fifo = util.Queue()
    closed = util.Counter()
    
    mask = np.zeros_like(board)

    for i in range(game_size):
        # White pieces on left edge
        if board[i, 0] == 1:
            mask[i, 0] = 1
            fifo.push((i, 0))

    while True:
            if fifo.isEmpty():
                return mask
            node = fifo.pop()
            if closed[node] == 0:
                closed[node] += 1
                west_neighbors = get_neighbors(board, node, 1)
                for child in west_neighbors:
                    mask[child] = 1
                    fifo.push(child)

# BFS for white pieces connected to east edge
def bfs_left(board, game_size):
    fifo = util.Queue()
    closed = util.Counter()
    
    mask = np.zeros_like(board)

    for i in range(game_size):
        # White pieces on left edge
        if board[i, game_size-1] == 1:
            mask[i, game_size-1] = 1
            fifo.push((i, game_size-1))

    while True:
            if fifo.isEmpty():
                return mask
            node = fifo.pop()
            if closed[node] == 0:
                closed[node] += 1
                east_neighbors = get_neighbors(board, node, 1)
                for child in east_neighbors:
                    mask[child] = 1
                    fifo.push(child)

# BFS for black pieces connected to north edge
def bfs_down(board, game_size):
    fifo = util.Queue()
    closed = util.Counter()
    
    mask = np.zeros_like(board)

    for i in range(game_size):
        # White pieces on left edge
        if board[0, i] == -1:
            mask[0, i] = -1
            fifo.push((0, i))

    while True:
            if fifo.isEmpty():
                return mask
            node = fifo.pop()
            if closed[node] == 0:
                closed[node] += 1
                south_neighbors = get_neighbors(board, node, -1)
                for child in south_neighbors:
                    mask[child] = -1
                    fifo.push(child)

# BFS for black pieces connected to south edge
def bfs_up(board, game_size):
    fifo = util.Queue()
    closed = util.Counter()
    
    mask = np.zeros_like(board)

    for i in range(game_size):
        # White pieces on left edge
        if board[game_size-1, i] == -1:
            mask[game_size-1, i] = -1
            fifo.push((game_size-1, i))

    while True:
            if fifo.isEmpty():
                return mask
            node = fifo.pop()
            if closed[node] == 0:
                closed[node] += 1
                north_neighbors = get_neighbors(board, node, -1)
                for child in north_neighbors:
                    mask[child] = -1
                    fifo.push(child)


