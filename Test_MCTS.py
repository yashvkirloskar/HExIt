from MCTS import *

batch_size = 2
mcts = MCTS(size=5, batch_size=batch_size, simulations_per_state=500, max_depth=4, apprentice=None)
inputData, distributions = mcts.generateExpertBatch()
testShapeInput = np.zeros((2, batch_size, 6, 9, 9))
testShapeDistributions = np.zeros((2, batch_size, 25))
assert(inputData.shape == testShapeInput.shape)
assert(distributions.shape == testShapeDistributions.shape)