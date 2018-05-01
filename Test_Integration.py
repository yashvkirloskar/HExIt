import numpy as np
from MCTS import MCTS
from Apprentice import Apprentice
from MCTS_utils import board_from_channels
import shutil
import time
import os
from State import State

def printActionDistribution(ad):
    for a in ad:
        if str(a)[0] == '-':
            print("%.2f" %(a * -1), " ", end='')
        else:
            print("%.2f" %a, " ", end='')
    print()


# Create data batch, feed it to apprentice, train, then predict, make sure shapes are correct

def testBasicIntegration(batch_size=3, game_size=5, simulations_per_state=500, threaded=False):
    if(os.path.exists("test_basic_integration")):
        shutil.rmtree("test_basic_integration")
    start = time.time()
    print("Testing basic Integration with batch_size = ", batch_size, ", simulations_per_state = ", simulations_per_state)

    num_actions = game_size**2

    mcts = MCTS(size=game_size, batch_size=batch_size, simulations_per_state=simulations_per_state,
        max_depth=4, apprentice=None, threaded=threaded)
    print("GENERATING EXPERT TRAINING BATCH")
    train_inputs, train_labels = mcts.generateExpertBatch()
    print("DONE GENERATING EXPERT TRAINING BATCH")

    apprentice = Apprentice(name="test_basic_integration", board_size=5, batch_size=mcts.num_threads)
    # not sure what mask does
    print("TRAINING APPRENTICE")
    apprentice.train(train_inputs, train_labels)
    print("DONE TRAINING APPRENTICE")

    # once train works, add a line to call apprentice.predict, and examine output shape.
    print("GENERATING EXPERT TEST BATCH")
    test_states, test_labels = mcts.generateExpertBatch()
    print("DONE GENERATING EXPERT TEST BATCH")
    print("PREDICTING LABELS FOR TEST BATCH")
    predicted_labels = apprentice.predict(test_states)
    print("DONE PREDICTING LABELS FOR TEST BATCH")

    assert(predicted_labels.shape == (2, batch_size, num_actions))

    # compare the apprentice-predicted output with the expert-generated output
    # we dont expect them to match too well after such little training, but just examine a few

    print("White to move:")
    print(board_from_channels(test_states[0]))
    print("Expert-Generated White Distribution for this state:")
    printActionDistribution(test_labels[0])
    print("Apprentice-Predicted White Distribution for this state:")
    printActionDistribution(predicted_labels[0][0])
    print("hopefully the distributions match")
    print()

    print("Black to move:")
    print(board_from_channels(test_states[batch_size]))
    print("Expert-Generated Black Distribution for this state:")
    printActionDistribution(test_labels[batch_size])
    print("Apprentice-Predicted White Distribution for this state:")
    printActionDistribution(predicted_labels[1][0])
    print("hopefully the distributions match")

    shutil.rmtree("test_basic_integration")
    end = time.time()
    print("Basic Integration test passed! Took", end - start, "seconds\n\n")


def testMultipleIterations(num_iterations=3, batch_size=2, game_size=5, simulations_per_state=50, threaded=False):
    if(os.path.exists("test_multiple_integration")):
        shutil.rmtree("test_multiple_integration")
    start = time.time()
    print("Testing", num_iterations, "iterations of Integration with batch_size = ", batch_size, ", simulations_per_state = ", simulations_per_state)

    num_actions = game_size ** 2

    mcts_initial = MCTS(size=game_size, batch_size=batch_size,
        simulations_per_state=simulations_per_state,
        max_depth=3, apprentice=None, threaded=threaded)
    apprentice = Apprentice(name="test_multiple_integration", board_size=game_size, batch_size=mcts_initial.num_threads)
    mcts_assisted = MCTS(size=game_size, batch_size=batch_size,
        simulations_per_state=simulations_per_state,
        max_depth=3, apprentice=apprentice, threaded=threaded)

    # first round of expert
    print("GENERATING INITIAL EXPERT BATCH")
    train_inputs, train_labels = mcts_initial.generateExpertBatch()
    print("FINSIHED GENERATING INITIAL EXPERT BATCH")
    # first round of apprentice
    print("TRAINING APPRENTICE ON INITIAL BATCH")
    apprentice.train(train_inputs, train_labels)
    print("FINISHED TRAINING INITIAL APPRENTICE")

    for i in range(1, num_iterations):
        print("GENERATING EXPERT BATCH FOR ITERATION", i)
        train_inputs, train_labels = mcts_assisted.generateExpertBatch()
        print("FINSIHED GENERATING EXPERT BATCH FOR ITERATION", i)
        print("TRAINING APPRENTICE FOR ITERATION", i)
        apprentice.train(train_inputs, train_labels)
        print("FINISHED TRAINING APPRENTICE FOR ITERATION ", i)


    # test our apprentice against the action distribution generated by the base (no apprentice help) MCTS
    print("GENERATING TEST BATCH")
    test_states, test_labels = mcts_initial.generateExpertBatch()
    print("FINISHED GENERATING TEST BATCH")
    print("PREDICTING LABELS FOR TEST BATCH")
    predicted_labels = apprentice.predict(test_states)
    print("FINISHED PREDICTING LABELS FOR TEST BATCH")
    assert(predicted_labels.shape == (2, batch_size, num_actions))

    # compare the apprentice-predicted output with the expert-generated output
    # we dont expect them to match too well after such little training, but just examine a few
    print("White to move:")
    print(board_from_channels(test_states[0]))
    print("Expert-Generated White Distribution for this state:")
    printActionDistribution(test_labels[0])
    print("Apprentice-Predicted White Distribution for this state:")
    printActionDistribution(predicted_labels[0][0])
    print("hopefully the distributions match")
    # make sure any nonzero entries are legal moves
    for i in range(batch_size):
        app_ad = predicted_labels[0][i]
        state = State(board_from_channels(test_states[i]))
        for j in range(num_actions):
            if app_ad[j] != 0:
                assert(state.isLegalAction(j))

    print()
        
    print("Black to move:")
    print(board_from_channels(test_states[batch_size]))
    print("Expert-Generated Black Distribution for this state:")
    printActionDistribution(test_labels[batch_size])
    print("Apprentice-Predicted White Distribution for this state:")
    printActionDistribution(predicted_labels[1][0])
    print("hopefully the distributions match")
    # make sure any nonzero entries are legal moves
    for i in range(batch_size):
        app_ad = predicted_labels[1][i]
        state = State(board_from_channels(test_states[batch_size + i]))
        for j in range(num_actions):
            if app_ad[j] != 0:
                assert(state.isLegalAction(j))


    shutil.rmtree("test_multiple_integration")
    end = time.time()
    print("Multiple Integration test passed! Took", end - start, "seconds\n\n")



def main():
    print ("Testing Integration...")
    overall_start = time.time()
    #testBasicIntegration(batch_size=1, simulations_per_state=10, threaded=True)
    testMultipleIterations(num_iterations=2, batch_size=16, simulations_per_state=10, threaded=True)
    #testMultipleIterations(num_iterations=2, batch_size=16, simulations_per_state=10, threaded=False)
    overall_end = time.time()
    print ("All tests passed! Took", overall_end - overall_start, "seconds\n\n")
    
    
if __name__ == '__main__':
    main()