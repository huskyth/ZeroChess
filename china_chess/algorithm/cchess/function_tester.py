from china_chess.algorithm.cchess.game_board import GameBoard
from china_chess.algorithm.cchess.mcts_tree import MCTS_tree

if __name__ == '__main__':
    gb = GameBoard()
    mcts = MCTS_tree(None, None)
    state, palyer = mcts.try_flip(gb.state, gb.current_player,
                                  mcts.is_black_turn(gb.current_player))
    action = GameBoard.get_legal_moves(gb.state, gb.current_player)[0]
    gb.state = GameBoard.sim_do_action(action, gb.state)
    gb.current_player = "w" if gb.current_player == "b" else "b"
    state, palyer = mcts.try_flip(gb.state, gb.current_player,
                                  mcts.is_black_turn(gb.current_player))
