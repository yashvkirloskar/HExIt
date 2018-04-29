from Apprentice import *
from MCTS import *
import shutil

def testTrain():
    print ("Starting testTrain")
    apprentice = Apprentice("testApprentice", 5, 2)
    mcts = MCTS(5, 2, 50, 3)
    batch, labels = mcts.generateExpertBatch()
    apprentice.train(batch, labels)
    shutil.rmtree("testApprentice")
    print ("Training occurred without incident, so test passed")

def testGetActionDistribution():
    print ("Starting testGetActionDistribution")
    apprentice = Apprentice("testApprentice", 5, 2)
    mcts = MCTS(5, 2, 50, 3)
    batch, labels = mcts.generateExpertBatch()
    apprentice.train(batch, labels)
    predicted = apprentice.predict(batch)
    print ("Printing out correct action distribution")
    print (np.sum(labels[0]))
    print (labels[0])
    print ("Printing out predicted action distribution")
    print (predicted[0])
    shutil.rmtree("testApprentice")
    print ("Look at outputs to judge whether passed")

def testGetActionDistributionSingle():
    print ("Starting testGetActionDistributionSingle")
    apprentice = Apprentice("testApprentice", 5, 2)
    mcts = MCTS(5, 1, 50, 3)
    batch, labels = mcts.generateExpertBatch()
    apprentice.train(batch, labels)
    print ("Predicting for player 1")
    predicted = apprentice.getActionDistributionSingle(batch[0], 1)
    print ("Printing out correct action distribution")
    print (labels[0])
    print (np.sum(labels[0]))
    print ("Printing out predicted action distribution")
    print (predicted[0])
    print ("Predicting for player 2")
    predicted = apprentice.getActionDistributionSingle(batch[1], -1)
    print ("Printing out correct action distribution")
    print (np.sum(labels[1]))
    print (labels[1])
    print ("Printing out predicted action distribution")
    print (predicted[0])
    shutil.rmtree("testApprentice")
    print ("Single actionDistibution passed")

def main():
    print ("Staring Apprentice Tests")
    testTrain()
    testGetActionDistribution()
    testGetActionDistributionSingle()
    print ("All tests passed")

if __name__ == '__main__':
    main()