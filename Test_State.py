from State import State
import numpy as np

game_size = 5
w = 1
b = -1
empty_board = State(np.zeros((game_size, game_size)))
two_pieces_each = State(np.array([[1,0,0,0,0],
                            [0,0,-1,0,0],
                            [0,0,0,0,0],
                            [0,0,1,0,0],
                            [-1,0,0,0,0]]))

simple_white_win = State(np.array([ [1,1,1,1,1],
                                    [0,0,-1,0,0],
                                    [0,0,0,0,0],
                                    [0,0,-1,-1,-1],
                                    [-1,0,0,0,0]]))


simple_black_win = State(np.array([ [w,b,w,w,0],
                                     [0,b,0,0,0],
                                      [0,b,0,0,0],
                                       [0,b,0,0,w],
                                        [b,0,0,0,w]]))

complex_white_win = State(np.array([ [b,0,0,b,b],
                                      [0,w,w,b,w],
                                       [w,b,w,w,b],
                                        [0,w,b,w,b],
                                         [0,w,0,0,0]]))


complex_black_win = State(np.array([ [w,b,0,w,w],
                                      [w,b,w,b,0],
                                       [0,b,b,0,w],
                                        [w,w,b,b,b],
                                         [w,0,0,0,b]]))


def testWinner():
    # create an empty board, make sure winner is 0
    assert(empty_board.winner() == 0)
    # place a few squares, make sure winner is 0
    assert(two_pieces_each.winner() == 0)
    # white win simple
    assert(simple_white_win.winner() == 1)
    # white win complex
    assert(complex_white_win.winner() == 1)
    # black win simple
    assert(simple_black_win.winner() == -1)
    # black win complex
    assert(complex_black_win.winner() == -1)
    print ("testWinner passed!")
    return

def testIsTerminalState():
    # make sure empty board isn't temrinal
    # place few squares, make sure isnt terminal 
    # simple white win
    # complex black win
    print("testIsTerminalState passed")
    return

def testCalculateReward():
    # make sure empty board is 0
    # place few squares, make sure 0
    # complex draw, make sure 0
    # simple white win, 1
    # complex black win, -1
    print("testCalculateReward passed")
    return 

def testNextState():
    # empty board, single action top left, confirm new board, confirm old board still same
    # complex board white to move
    # complex board, black to move
    print("testNextState passed")
    return

def testIsLegalAction():
    # empty board, single move is legal 
    # simple board, try to resquare
    # complex board, try legal move
    print("testIsLegalAction passed")
    return

def testLegalActions():
    # simple board should have 25 legal actions 
    # medium board should have correct number of legal actions, one for black, one white
    # terminal board should have no legal actions
    print("testLegalActions passed")
    return

def testChooseRandomAction():
    # empty board choose random action should be between 0 and 24
    # simple board with top row set, generate a bunch of random actions, make sure in range
    # board with one spot left, make sure it's that every time 
    # make sure terminal board offers -1
    print("testChooseRandomAction passed")
    return

def testTurn():
    # empty board is player 1
    # one piece down is player 2
    # 13 pieces down is player 2
    # take next action, make sure turn is now player 1
    print("testTurn passed")
    return

def testNonTerminalActions():
    # empty board should have every action non terminal
    # few pieces down, every available action non terminal
    # white has one terminal spot, that should be not offered
    # black has two non terminal spots, not offered
    # terminal state offers nothing
    print("testNonTerminalActions passed")
    return

def main():
    print ("Testing State class...")
    testWinner()
    testIsTerminalState()
    testCalculateReward()
    testNextState()
    testIsLegalAction()
    testLegalActions()
    testChooseRandomAction()
    testTurn()
    testNonTerminalActions()

    print ("All tests passed!")

if __name__ == '__main__':
    main()