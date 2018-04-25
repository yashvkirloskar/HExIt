from State import State
import numpy as np

game_size = 5
w = 1
b = -1
empty_board = State(np.zeros((game_size, game_size)))

top_left = State(np.array([[w,0,0,0,0],
                            [0,0,0,0,0],
                             [0,0,0,0,0],
                              [0,0,0,0,0],
                               [0,0,0,0,0]]))

two_pieces_each = State(np.array([[1,0,0,0,0],
                                   [0,0,-1,0,0],
                                    [0,0,0,0,0],
                                     [0,0,1,0,0],
                                      [-1,0,0,0,0]]))

three_two = State(np.array([      [1,0,0,0,0],
                                   [0,0,-1,0,0],
                                    [0,0,0,0,0],
                                     [0,0,1,0,0],
                                      [-1,0,0,0,1]]))

three_two_legal_actions = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23]


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

complex_draw_black_move = State(np.array([[w,b,0,w,w],
                                           [w,0,w,b,0],
                                            [0,b,b,0,w],
                                             [w,w,b,b,b],
                                              [w,0,0,0,b]]))

complex_draw_black_move_nta = [2,9,10,13,21,22,23]

complex_draw_white_move = State(np.array([[w,b,0,w,w],
                                           [w,0,w,b,0],
                                            [0,b,b,0,w],
                                             [w,w,b,b,b],
                                              [w,b,0,0,b]]))

complex_draw_white_move_options = [2,6,9,10,13,22,23]

test_channels = np.array([[w,b,w,b,b],
                           [b,w,w,0,b],
                            [w,w,b,w,b],
                             [b,w,0,0,w],
                              [b,b,b,w,w]])

expected_channels = np.zeros((6, 9, 9))
expected_channels[0] = np.array([[w,w,0,0,0,0,0,w,w],
                                  [w,w,0,0,0,0,0,w,w],
                                   [w,w,w,0,w,0,0,w,w],
                                    [w,w,0,w,w,0,0,w,w],
                                     [w,w,w,w,0,w,0,w,w],
                                      [w,w,0,w,0,0,w,w,w],
                                       [w,w,0,0,0,w,w,w,w],
                                        [w,w,0,0,0,0,0,w,w],
                                         [w,w,0,0,0,0,0,w,w]])
expected_channels[1] = np.array([[b,b,b,b,b,b,b,b,b],
                                  [b,b,b,b,b,b,b,b,b],
                                   [0,0,0,b,0,b,b,0,0],
                                    [0,0,b,0,0,0,b,0,0],
                                     [0,0,0,0,b,0,b,0,0],
                                      [0,0,b,0,0,0,0,0,0],
                                       [0,0,b,b,b,0,0,0,0],
                                        [b,b,b,b,b,b,b,b,b],
                                         [b,b,b,b,b,b,b,b,b]])
expected_channels[2] = np.array([[w,w,0,0,0,0,0,0,0],
                                  [w,w,0,0,0,0,0,0,0],
                                   [w,w,w,0,w,0,0,0,0],
                                    [w,w,0,w,w,0,0,0,0],
                                     [w,w,w,w,0,0,0,0,0],
                                      [w,w,0,w,0,0,0,0,0],
                                       [w,w,0,0,0,0,0,0,0],
                                        [w,w,0,0,0,0,0,0,0],
                                         [w,w,0,0,0,0,0,0,0]])
expected_channels[3] = np.array([[0,0,0,0,0,0,0,w,w],
                                  [0,0,0,0,0,0,0,w,w],
                                   [0,0,0,0,0,0,0,w,w],
                                    [0,0,0,0,0,0,0,w,w],
                                     [0,0,0,0,0,0,0,w,w],
                                      [0,0,0,0,0,0,w,w,w],
                                       [0,0,0,0,0,w,w,w,w],
                                        [0,0,0,0,0,0,0,w,w],
                                         [0,0,0,0,0,0,0,w,w]])
expected_channels[4] = np.array([[b,b,b,b,b,b,b,b,b],
                                  [b,b,b,b,b,b,b,b,b],
                                   [0,0,0,b,0,b,b,0,0],
                                    [0,0,b,0,0,0,b,0,0],
                                     [0,0,0,0,0,0,b,0,0],
                                      [0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0],
                                         [0,0,0,0,0,0,0,0,0]])
expected_channels[5] = np.array([[0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0],
                                     [0,0,0,0,0,0,0,0,0],
                                      [0,0,b,0,0,0,0,0,0],
                                       [0,0,b,b,b,0,0,0,0],
                                        [b,b,b,b,b,b,b,b,b],
                                         [b,b,b,b,b,b,b,b,b]])
expected_channels = np.fabs(expected_channels)
channelTest = State(test_channels)


test_channels_win_white = np.array([[0,b,b,w,w],
                                     [b,w,w,b,b],
                                      [w,w,b,w,b],
                                       [b,w,0,0,w],
                                        [b,b,b,w,w]])

expected_channels_win_white = np.zeros((6, 9, 9))
expected_channels_win_white[0] = np.array([[w,w,0,0,0,0,0,w,w],
                                            [w,w,0,0,0,0,0,w,w],
                                             [w,w,0,0,0,w,w,w,w],
                                              [w,w,0,w,w,0,0,w,w],
                                               [w,w,w,w,0,w,0,w,w],
                                                [w,w,0,w,0,0,w,w,w],
                                                 [w,w,0,0,0,w,w,w,w],
                                                  [w,w,0,0,0,0,0,w,w],
                                                   [w,w,0,0,0,0,0,w,w]])
expected_channels_win_white[1] = np.array([[b,b,b,b,b,b,b,b,b],
                                            [b,b,b,b,b,b,b,b,b],
                                             [0,0,0,b,b,0,0,0,0],
                                              [0,0,b,0,0,b,b,0,0],
                                               [0,0,0,0,b,0,b,0,0],
                                                [0,0,b,0,0,0,0,0,0],
                                                 [0,0,b,b,b,0,0,0,0],
                                                  [b,b,b,b,b,b,b,b,b],
                                                   [b,b,b,b,b,b,b,b,b]])
expected_channels_win_white[2] = np.array([[w,w,0,0,0,0,0,w,w],
                                            [w,w,0,0,0,0,0,w,w],
                                             [w,w,0,0,0,w,w,w,w],
                                              [w,w,0,w,w,0,0,w,w],
                                               [w,w,w,w,0,0,0,w,w],
                                                [w,w,0,w,0,0,w,w,w],
                                                 [w,w,0,0,0,w,w,w,w],
                                                  [w,w,0,0,0,0,0,w,w],
                                                   [w,w,0,0,0,0,0,w,w]])
expected_channels_win_white[3] = np.array([[w,w,0,0,0,0,0,w,w],
                                            [w,w,0,0,0,0,0,w,w],
                                             [w,w,0,0,0,w,w,w,w],
                                              [w,w,0,w,w,0,0,w,w],
                                               [w,w,w,w,0,0,0,w,w],
                                                [w,w,0,w,0,0,w,w,w],
                                                 [w,w,0,0,0,w,w,w,w],
                                                  [w,w,0,0,0,0,0,w,w],
                                                   [w,w,0,0,0,0,0,w,w]])
expected_channels_win_white[4] = np.array([[b,b,b,b,b,b,b,b,b],
                                            [b,b,b,b,b,b,b,b,b],
                                             [0,0,0,b,b,0,0,0,0],
                                              [0,0,b,0,0,0,0,0,0],
                                               [0,0,0,0,0,0,0,0,0],
                                                [0,0,0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0,0,0],
                                                  [0,0,0,0,0,0,0,0,0],
                                                   [0,0,0,0,0,0,0,0,0]])
expected_channels_win_white[5] = np.array([[0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0],
                                              [0,0,0,0,0,0,0,0,0],
                                               [0,0,0,0,0,0,0,0,0],
                                                [0,0,b,0,0,0,0,0,0],
                                                 [0,0,b,b,b,0,0,0,0],
                                                  [b,b,b,b,b,b,b,b,b],
                                                   [b,b,b,b,b,b,b,b,b]])
expected_channels_win_white = np.fabs(expected_channels_win_white)
channelTest_win_white = State(test_channels_win_white)


test_channels_win_black = np.array([[w,0,b,w,w],
                                     [b,w,b,b,b],
                                      [w,w,b,w,0],
                                       [b,b,w,0,b],
                                        [b,b,w,w,w]])

expected_channels_win_black = np.zeros((6, 9, 9))
expected_channels_win_black[0] = np.array([[w,w,0,0,0,0,0,w,w],
                                            [w,w,0,0,0,0,0,w,w],
                                             [w,w,w,0,0,w,w,w,w],
                                              [w,w,0,w,0,0,0,w,w],
                                               [w,w,w,w,0,w,0,w,w],
                                                [w,w,0,0,w,0,0,w,w],
                                                 [w,w,0,0,w,w,w,w,w],
                                                  [w,w,0,0,0,0,0,w,w],
                                                   [w,w,0,0,0,0,0,w,w]])
expected_channels_win_black[1] = np.array([[b,b,b,b,b,b,b,b,b],
                                            [b,b,b,b,b,b,b,b,b],
                                             [0,0,0,0,b,0,0,0,0],
                                              [0,0,b,0,b,b,b,0,0],
                                               [0,0,0,0,b,0,0,0,0],
                                                [0,0,b,b,0,0,b,0,0],
                                                 [0,0,b,b,0,0,0,0,0],
                                                  [b,b,b,b,b,b,b,b,b],
                                                   [b,b,b,b,b,b,b,b,b]])
expected_channels_win_black[2] = np.array([[w,w,0,0,0,0,0,0,0],
                                            [w,w,0,0,0,0,0,0,0],
                                             [w,w,w,0,0,0,0,0,0],
                                              [w,w,0,w,0,0,0,0,0],
                                               [w,w,w,w,0,0,0,0,0],
                                                [w,w,0,0,0,0,0,0,0],
                                                 [w,w,0,0,0,0,0,0,0],
                                                  [w,w,0,0,0,0,0,0,0],
                                                   [w,w,0,0,0,0,0,0,0]])
expected_channels_win_black[3] = np.array([[0,0,0,0,0,0,0,w,w],
                                            [0,0,0,0,0,0,0,w,w],
                                             [0,0,0,0,0,w,w,w,w],
                                              [0,0,0,0,0,0,0,w,w],
                                               [0,0,0,0,0,w,0,w,w],
                                                [0,0,0,0,w,0,0,w,w],
                                                 [0,0,0,0,w,w,w,w,w],
                                                  [0,0,0,0,0,0,0,w,w],
                                                   [0,0,0,0,0,0,0,w,w]])
expected_channels_win_black[4] = np.array([[b,b,b,b,b,b,b,b,b],
                                            [b,b,b,b,b,b,b,b,b],
                                             [0,0,0,0,b,0,0,0,0],
                                              [0,0,0,0,b,b,b,0,0],
                                               [0,0,0,0,b,0,0,0,0],
                                                [0,0,b,b,0,0,0,0,0],
                                                 [0,0,b,b,0,0,0,0,0],
                                                  [b,b,b,b,b,b,b,b,b],
                                                   [b,b,b,b,b,b,b,b,b]])
expected_channels_win_black[5] = np.array([[b,b,b,b,b,b,b,b,b],
                                            [b,b,b,b,b,b,b,b,b],
                                             [0,0,0,0,b,0,0,0,0],
                                              [0,0,0,0,b,b,b,0,0],
                                               [0,0,0,0,b,0,0,0,0],
                                                [0,0,b,b,0,0,0,0,0],
                                                 [0,0,b,b,0,0,0,0,0],
                                                  [b,b,b,b,b,b,b,b,b],
                                                   [b,b,b,b,b,b,b,b,b]])

expected_channels_win_black = np.fabs(expected_channels_win_black)
channelTest_win_black = State(test_channels_win_black)

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
    print ("testWinner passed")
    return

def testIsTerminalState():
    # make sure empty board isn't temrinal
    assert(not empty_board.isTerminalState())
    # place few squares, make sure isnt terminal
    assert(not two_pieces_each.isTerminalState()) 
    # complex draw
    assert(not complex_draw_black_move.isTerminalState())
    # simple white win
    assert(simple_white_win.isTerminalState())
    # complex black win
    assert(complex_black_win.isTerminalState())
    print("testIsTerminalState passed")
    return

def testCalculateReward():
    # make sure empty board is 0
    assert(empty_board.calculateReward() == 0)
    # place few squares, make sure 0
    assert(two_pieces_each.calculateReward() == 0)
    # complex draw, make sure 0
    assert(complex_draw_black_move.calculateReward() == 0)
    # simple white win, 1
    assert(simple_white_win.calculateReward() == 1)
    # complex black win, -1
    assert(complex_black_win.calculateReward() == -1)
    print("testCalculateReward passed")
    return 

def testNextState():
    # empty board, single action top left, confirm new board, confirm old board still same
    single_move = empty_board.nextState(0)
    assert(single_move == top_left)
    # simple board white to move
    white_move = two_pieces_each.nextState(24)
    assert(white_move == three_two)
    # complex board, black to move
    draw = complex_draw_black_move.nextState(21)
    assert(draw == complex_draw_white_move)

    print("testNextState passed")
    return

def testIsLegalAction():
    # empty board, single move is legal
    assert(empty_board.isLegalAction(17))
    # simple board, try to resquare
    assert(not two_pieces_each.isLegalAction(7))
    # complex board, try legal move
    assert(complex_draw_white_move.isLegalAction(9))
    print("testIsLegalAction passed")
    return

def testLegalActions():
    # simple board should have 25 legal actions 
    assert(len(empty_board.legalActions()) == 25)
    # medium board should have correct number of legal actions, one for black, one white
    assert(len(three_two.legalActions()) == 20)
    assert(len(complex_draw_white_move.legalActions()) == 7)
    # terminal board should have no legal actions
    assert(len(complex_white_win.legalActions()) == 0)
    print("testLegalActions passed")
    return

def testChooseRandomAction():
    # empty board choose random action should be between 0 and 24
    assert(empty_board.chooseRandomAction() in range(25))
    # simple board with top row set, generate a bunch of random actions, make sure in range
    for i in range(20):
        assert(three_two.chooseRandomAction() in three_two_legal_actions)
    # board with one spot left, make sure it's that every time 
    for i in range(20):
        assert(complex_draw_white_move.chooseRandomAction() in complex_draw_white_move_options)
    # make sure terminal board offers -1
    for i in range(5):
        assert(complex_black_win.chooseRandomAction() == -1)
    print("testChooseRandomAction passed")
    return

def testTurn():
    # empty board is player 1
    assert(empty_board.turn() == 1 and empty_board.isPlayerOneTurn() and not empty_board.isPlayerTwoTurn())
    # one piece down is player 2
    assert(three_two.turn() == -1 and not three_two.isPlayerOneTurn() and three_two.isPlayerTwoTurn())
    # 18 pieces down is player 1
    assert(complex_draw_white_move.turn() == 1 and complex_draw_white_move.isPlayerOneTurn()
        and not complex_draw_white_move.isPlayerTwoTurn())
    # take next action, make sure turn is now player 2
    black_move = complex_draw_white_move.nextState(9)
    assert(black_move.turn() == -1 and not black_move.isPlayerOneTurn() and black_move.isPlayerTwoTurn())
    print("testTurn passed")
    return

def testNonTerminalActions():
    # empty board should have every action non terminal
    assert(len(empty_board.nonTerminalActions()) == 25)
    # few pieces down, every available action non terminal
    assert(three_two.nonTerminalActions() == three_two_legal_actions)
    # black has only certain non terminal spots
    assert(complex_draw_black_move.nonTerminalActions() == complex_draw_black_move_nta)
    assert(complex_draw_black_move.nonTerminalActions() != complex_draw_black_move.legalActions())
    # terminal state offers nothing
    assert(complex_black_win.nonTerminalActions() == [])
    print("testNonTerminalActions passed")
    return

def testChannelsFromState():
    test = channelTest.channels_from_state()
    for i in range(6):
        
        assert(np.array_equal(test[i], expected_channels[i]))

    test_win_white = channelTest_win_white.channels_from_state()
    for i in range(6):
        assert(np.array_equal(test_win_white[i], expected_channels_win_white[i]))

    test_win_black = channelTest_win_black.channels_from_state()
    for i in range(6):
        assert(np.array_equal(test_win_black[i], expected_channels_win_black[i]))
    print ("testChannelsFromState passed")


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
    testChannelsFromState()
    print ("All tests passed!")

if __name__ == '__main__':
    main()