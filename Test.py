from VectorHex import *
from GameUtils import *

testGame = VectorHex(5)
testGame.player_move([0, 1])
assert(testGame.board[0, 1] == 1)
for neighbor in get_neighbors(testGame.board, (0, 1), testGame.turn):
    assert(testGame.board[neighbor] == -1 or testGame.board[neighbor] == 0)
testGame.player_move([0,0])
testGame.player_move([1,1])
testGame.player_move([1,0])
testGame.player_move([2,1])
testGame.player_move([2,0])
testGame.player_move([3,1])
testGame.player_move([3,0])
testGame.player_move([4,1])
testGame.player_move([4,0])
assert(check_win(testGame.board, testGame.turn, testGame.game_size) == "Win")
assert(np.array_equal(testGame.board, board_from_channels(testGame.vector)))