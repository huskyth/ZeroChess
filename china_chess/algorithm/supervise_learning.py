from pathlib import Path
import os.path
import sys

path = str(Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(path)
print(sys.path)

from china_chess.algorithm.data_loader import LoadData

from china_chese_game import *
from china_chess.algorithm.icy_chess.game_state import GameState
from china_chess.algorithm.sl_net import NNetWrapper
from china_chess.algorithm.icy_chess.game_convert import *


def left_right_move_flip(origin_move):
    temp = REVERSE_LETTERS[LETTERS_TO_IND[origin_move[0]]] + origin_move[1] + REVERSE_LETTERS[
        LETTERS_TO_IND[origin_move[2]]] + origin_move[3]
    return temp


def top_bottom_move_flip(origin_move):
    return origin_move[0] + REVERSE_NUMBERS[NUMBERS_TO_IND[origin_move[1]]] + origin_move[2] + REVERSE_NUMBERS[
        NUMBERS_TO_IND[origin_move[3]]]


def expand_data(board, game_state):
    # TODO://可能有问题
    data, pi, v, player = board
    formal_data = data
    num = np.argmax(pi)
    move = LABELS[num]
    if player == 'b':
        move = top_bottom_move_flip(move)
    if player == 'r':
        move = left_right_move_flip(move)
    origin_move = LABELS[num]
    print_one_turn(data, move, game_state, v, origin_move)
    pi = [0] * len(LABELS)
    pi[LABELS_TO_INDEX[move]] = 1
    if player == 'r':
        formal_data = data[::-1, :]
    data = np.array(integer_to_state_str(formal_data))
    data = boardarr2netinput(data, 'w' if player == 'r' else 'b')

    return data, pi, v, player


# after expand_data
def flip_all(data, pi, v, player, ano_gs):
    formal_data = data[:, ::-1]
    num = np.argmax(pi)
    move = left_right_move_flip(LABELS[num])
    pi = [0] * len(LABELS)
    pi[LABELS_TO_INDEX[move]] = 1

    print_one_turn(formal_data, move, ano_gs, v, LABELS[num])
    data = np.array(integer_to_state_str(formal_data))
    data = boardarr2netinput(data, 'w' if player == 'r' else 'b')
    return data, pi, v, player


def print_one_turn(current_board, move, state, v, origin_move):
    return None
    ccb = ChinaChessBoard()
    ccb.to_chess_map(current_board)
    print("before from example，move is {}, v = {}, 原始的行为为{}".format(move, v, origin_move))
    ccb.print_visible_string(is_print_screen=True)
    print("before from state，move is {}, v = {},原始的行为为{}".format(move, v, origin_move, ))
    state.display()
    print("after from move，move is {}, v = {},原始的行为为{}".format(move, v, origin_move))
    state.do_move(move)
    state.display()
    print()


def main():
    nn = NNetWrapper()
    examples = LoadData().get_all_examples()
    gs = GameState()
    ano_gs = GameState()
    for e_i in range(len(examples)):
        board = examples[e_i]
        examples[e_i] = expand_data(board, gs)
        data, pi, v, player = examples[e_i]
        flipped = flip_all(board[0], pi, v, player, ano_gs)
        examples.append(flipped)

    nn.load_checkpoint(filename="best_loss.pth.tar")
    nn.train(examples, 0)


if __name__ == "__main__":
    main()
