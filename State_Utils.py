# Utility functions for checking game conditions
import util
import numpy as np
from GameUtils import get_neighbors

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
                neighbors = get_neighbors(board, node, 1)
                west_neighbors = neighbors[0:2]
                for child in west_neighbors:
                    mask[child] = 1
                    fifo.push(child)