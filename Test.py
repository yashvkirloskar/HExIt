from Hex import *

testGame = Hex(3)
testGame.player_move([0, 1])
assert(testGame.board[0, 1].player == 0)
for neighbor in testGame.board[0, 1].get_neighbors():
    assert(isinstance(neighbor, EmptyPiece))
testGame.player_move([0,0])
testGame.player_move([1,1])
testGame.player_move([1,0])
testGame.player_move([2,1])
testGame.player_move([2,0])
testGame.player_move([3,1])
testGame.player_move([3,0])
testGame.player_move([4,1])
testGame.player_move([4,0])
assert(testGame.check_win() == "Win")