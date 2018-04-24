from MCTS_utils import *
import numpy as np

a = [0 for i in range(100)]
for i in range(100):
    a[i] = generateRandomState(5, 1)
    assert(a[i].turn() == 1)

for i in range(100):
    a[i] = generateRandomState(5, -1)
    assert(a[i].turn() == -1)