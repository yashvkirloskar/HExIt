from MCTS import *
from Apprentice import *
import shutil

batch_size = 2
apprentice = Apprentice("testParallel", 5, batch_size)
# mcts_initial = MCTS(size=5, batch_size=batch_size, simulations_per_state=500, max_depth=4, apprentice=None, parallel=True)
# batch, labels = mcts_initial.generateExpertBatch()
# apprentice.train(batch, labels)
mcts = MCTS(size=5, batch_size=batch_size, simulations_per_state=500, max_depth=4, apprentice=apprentice, parallel=True)
parallelBatch, labels = mcts.generateExpertBatch()
print (label.shape)
shutil.rmtree("testParallel")