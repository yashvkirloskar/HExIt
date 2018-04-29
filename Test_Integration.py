import numpy as np
from MCTS import MCTS
from Apprentice import Apprentice
import shutil


# Create data batch, feed it to apprentice, train, then predict, make sure shapes are correct
def testBasicIntegration():
    batch_size = 1
    mcts = MCTS(size=5, batch_size=batch_size, simulations_per_state=500, max_depth=4, apprentice=None)
    train_inputs, train_labels = mcts.generateExpertBatch()

    apprentice = Apprentice(name="test_basic_integration", board_size=5, batch_size=batch_size)
    # not sure what mask does
    apprentice.train(train_inputs, train_labels) # broken

    # once train works, add a line to call apprentice.predict, and examine output shape.

def main():
    print ("Testing Integration...")
    testBasicIntegration()
    print ("All tests passed!")
    shutil.rmtree("test_basic_integration")

if __name__ == '__main__':
    main()