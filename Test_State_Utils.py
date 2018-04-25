from State_Utils import *

def testBFSRight(board, game_size, expected):
    test = bfs_right(board, game_size)
    assert(np.array_equal(test, expected))

def testBFSLeft(board, game_size, expected):
    test = bfs_left(board, game_size)
    assert(np.array_equal(test, expected))

def testBFSDown(board, game_size, expected):
    test = bfs_down(board, game_size)
    assert(np.array_equal(test, expected))

def testBFSUp(board, game_size, expected):
    test = bfs_up(board, game_size)
    assert(np.array_equal(test, expected))

w = 1
b = -1

test = np.array([[w,b,w,b,0],
                  [b,w,w,b,b],
                   [w,w,b,b,w],
                    [b,w,0,0,w],
                     [b,b,b,w,w]])

expectedRight = np.array([[w,0,w,0,0],
                          [0,w,w,0,0],
                           [w,w,0,0,0],
                            [0,w,0,0,0],
                             [0,0,0,0,0]])

expectedLeft = np.array([[0,0,0,0,0],
                           [0,0,0,0,0],
                            [0,0,0,0,w],
                             [0,0,0,0,w],
                              [0,0,0,w,w]])

expectedDown = np.array([[0,b,0,b,0],
                          [b,0,0,b,b],
                           [0,0,b,b,0],
                            [0,0,0,0,0],
                             [0,0,0,0,0]])

expectedUp = np.array([[0,0,0,0,0],
                        [0,0,0,0,0],
                         [0,0,0,0,0],
                          [b,0,0,0,0],
                           [b,b,b,0,0]])

def main():
    print ("Testing State Utility Functions")
    testBFSRight(test, 5, expectedRight)
    testBFSLeft(test, 5, expectedLeft)
    testBFSDown(test, 5, expectedDown)
    testBFSUp(test, 5, expectedUp)
    print ("All tests passed.")

if __name__ == '__main__':
    main()