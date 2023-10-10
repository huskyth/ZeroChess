import os
from multiprocessing import Pipe, Process
from pickle import Pickler, Unpickler

from Coach import CloudpickleWrapper
from china_chess.algorithm.mcts_async import *
from china_chess.algorithm.mcts_origin import *
from china_chess.constant import countpiece, LABELS, LABELS_TO_INDEX
from china_chess.algorithm.sl_net import NNetWrapper as nn

TOP_TRAIN_NUMBER = 3
nnet = nn()


def save_train_examples(iteration, train_examples):
    folder = "training_examples"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, str(iteration) + ".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(train_examples)


def load_train_examples(example_path):
    with open(example_path, "rb") as f:
        train_example = Unpickler(f).load()
    return train_example


def find_top_examples():
    return []


def execute_episode(mcts):
    gs = GameState()
    train_examples = []
    episode_step = 0

    peace_round = 0
    remain_piece = countpiece(gs.state_str)
    mcts.update_with_move(-1)

    while True:
        episode_step += 1
        temp = int(episode_step < 10)

        move = mcts.get_move_probs(gs, temp=temp)
        pi = [0] * len(LABELS)
        pi[LABELS_TO_INDEX[move]] = 1
        bb = BaseChessBoard(gs.state_str)
        state_str = bb.get_board_arr()
        net_x = boardarr2netinput(state_str, gs.get_current_player())
        train_examples.append([net_x, pi, None, gs.get_current_player()])
        gs.do_move(move)
        is_end, winner, info = gs.game_end()
        mcts.update_with_move(move)

        remain_piece_round = countpiece(gs.state_str)
        if remain_piece_round < remain_piece:
            remain_piece = remain_piece_round
            peace_round = 0
        else:
            peace_round += 1

        if episode_step > 150 and peace_round > 60:
            for t in range(len(train_examples)):
                train_examples[t][2] = 0
            return train_examples

        if is_end:
            for t in range(len(train_examples)):
                if winner == gs.get_current_player():
                    train_examples[t][2] = 1
                else:
                    train_examples[t][2] = -1
            return train_examples


def eval(play1, random_player):
    players = [play1, None, random_player]
    cur_player = 1
    it = 0
    winner = None
    game_state = GameState()

    peace_round = 0
    remain_piece = countpiece(game_state.state_str)

    while not game_state.game_end()[0]:
        current_player = players[cur_player + 1]
        another_player = players[1 - cur_player]
        it += 1

        move = current_player.get_move_probs(game_state)
        game_state.do_move(move)
        is_end, winner, info = game_state.game_end()
        current_player.update_with_move(move)
        another_player.update_with_move(move)
        cur_player *= -1

        remain_piece_round = countpiece(game_state.state_str)
        if remain_piece_round < remain_piece:
            remain_piece = remain_piece_round
            peace_round = 0
        else:
            peace_round += 1

        if it > 150 and peace_round > 60:
            current_player.update_with_move(-1)
            another_player.update_with_move(-1)
            return None

        if is_end:
            current_player.update_with_move(-1)
            another_player.update_with_move(-1)
            return winner
    players[0].update_with_move(-1)
    players[-1].update_with_move(-1)
    return winner


def handle(parent_remote, worker_remote, mcts):
    parent_remote.close()
    save_train_iter = 0
    while True:
        cmd, data = worker_remote.recv()
        print(cmd, data)

        if cmd == "data_mining":
            train_examples = execute_episode(mcts)
            save_train_examples(save_train_iter, train_examples)
            save_train_iter += 1
        elif cmd == "train":
            # 主进程
            while True:
                break
                pass
            train_example_path = ''
            train_examples = []
            if len(os.listdir(train_example_path)) >= TOP_TRAIN_NUMBER:
                file_names = find_top_examples()
                for file in file_names:
                    train_examples += load_train_examples(file)
                nnet.train(train_examples)
        elif cmd == "predict":
            worker_remote.send(nnet.predict(data))


def multiprocess_handle():
    max_parallel = 10
    train_process_num = 1
    max_parallel += train_process_num
    parent_remote, worker_remote = zip(*[Pipe() for x in range(max_parallel)])
    mcts_list = [CloudpickleWrapper(MCTS()) for i in range(max_parallel)]

    process_list = [Process(target=handle, args=(p, w, m), daemon=True) for p, w, m in
                    zip(parent_remote, worker_remote, mcts_list)]

    for i in range(max_parallel):
        process_list[i].start()

    for i in range(max_parallel):
        worker_remote[i].close()

    for i in process_list:
        i.join()


if __name__ == '__main__':
    multiprocess_handle()
