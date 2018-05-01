from MCTS import *
from Apprentice import *
import shutil
import time

batch_size = 16
apprentice = Apprentice("testParallel", 5, batch_size)


start = time.time()
mcts_initial = MCTS(size=5, batch_size=batch_size, simulations_per_state=10, max_depth=4, apprentice=None, parallel=False)
batch, labels = mcts_initial.generateExpertBatch()
apprentice.train(batch, labels)
end = time.time()
print (end-start)

start = time.time()
mcts_initial = MCTS(size=5, batch_size=batch_size, simulations_per_state=10, max_depth=4, apprentice=None, parallel=True)
batch, labels = mcts_initial.generateExpertBatch()
end = time.time()
print (end-start)

# apprentice.train(batch, labels)
# mcts = MCTS(size=5, batch_size=batch_size, simulations_per_state=10, max_depth=4, apprentice=apprentice, parallel=True)
# parallelBatch, labels = mcts.generateExpertBatch()
# apprentice.train(parallelBatch, labels)
# print (labels.shape)
# shutil.rmtree("testParallel")
