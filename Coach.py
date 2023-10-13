import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
import random

from china_chess.algorithm.cchess.const_function import label2i, is_kill_move, labels_len
from china_chess.algorithm.cchess.mcts_tree import MCTS_tree

from china_chess.algorithm.mcts_async import *
from china_chess.algorithm.tensor_board_tool import MySummary
from china_chess.constant import LABELS, LABELS_TO_INDEX, countpiece
from china_chess.algorithm.sl_net import NNetWrapper as PolicyValueNetwork

log = logging.getLogger(__name__)


class Coach:

    def __init__(self, playout=400, in_search_threads=16, in_batch_size=512, exploration=True):
        self.policy_value_network = PolicyValueNetwork()
        self.buffer_size = 10000
        self.temperature = 1  # 1e-8    1e-3
        self.playout_counts = playout  # 400    #800    #1600    200
        self.epochs = 5
        self.kl_targ = 0.025
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.learning_rate = 0.001  # 5e-3    #    0.001

        self.summary = MySummary("elo")
        self.batch_size = in_batch_size  # 128    #512
        self.exploration = exploration

        self.data_buffer = deque(maxlen=self.buffer_size)
        self.game_board = GameBoard()
        self.search_threads = in_search_threads

        self.mcts = MCTS_tree(self.game_board.state, self.policy_value_network.predict, self.search_threads)

    def get_action(self, state, temperature=1e-3):
        # for i in range(self.playout_counts):
        #     state_sim = copy.deepcopy(state)
        #     self.mcts.do_simulation(state_sim, self.game_board.current_player, self.game_board.restrict_round)

        self.mcts.main(state, self.game_board.current_player, self.game_board.restrict_round, self.playout_counts)

        actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
        actions, visits = zip(*actions_visits)
        probs = softmax(1.0 / temperature * np.log(visits))  # + 1e-10
        move_probs = [[actions, probs]]

        if self.exploration:
            act = np.random.choice(actions, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
        else:
            act = np.random.choice(actions, p=probs)

        win_rate = self.mcts.Q(act)  # / 2.0 + 0.5
        self.mcts.update_tree(act)

        # if position.n < 30:    # self.top_steps
        #     move = select_weighted_random(position, on_board_move_prob)
        # else:
        #     move = select_most_likely(position, on_board_move_prob)

        return act, move_probs, win_rate

    def execute_episode(self):
        # self_play function
        self.game_board.reload()
        # p1, p2 = self.game_board.players
        states, mcts_probs, current_players = [], [], []
        z = None
        game_over = False
        winnner = ""
        start_time = time.time()
        # self.game_board.print_board(self.game_board.state)
        while not game_over:
            action, probs, win_rate = self.get_action(self.game_board.state, self.temperature)
            state, player = self.mcts.try_flip(self.game_board.state, self.game_board.current_player,
                                               self.mcts.is_black_turn(self.game_board.current_player))
            states.append(state)
            prob = np.zeros(labels_len)
            if self.mcts.is_black_turn(self.game_board.current_player):
                for idx in range(len(probs[0][0])):
                    # probs[0][0][idx] = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
                    act = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
                    # for idx in range(len(mcts_prob[0][0])):
                    prob[label2i[act]] = probs[0][1][idx]
            else:
                for idx in range(len(probs[0][0])):
                    prob[label2i[probs[0][0][idx]]] = probs[0][1][idx]
            mcts_probs.append(prob)
            # mcts_probs.append(probs)
            current_players.append(self.game_board.current_player)

            last_state = self.game_board.state
            # print(self.game_board.current_player, " now take a action : ", action, "[Step {}]".format(self.game_board.round))
            self.game_board.state = GameBoard.sim_do_action(action, self.game_board.state)
            self.game_board.round += 1
            self.game_board.current_player = "w" if self.game_board.current_player == "b" else "b"
            if is_kill_move(last_state, self.game_board.state) == 0:
                self.game_board.restrict_round += 1
            else:
                self.game_board.restrict_round = 0

            # self.game_board.print_board(self.game_board.state, action)

            if self.game_board.state.find('K') == -1 or self.game_board.state.find('k') == -1:
                z = np.zeros(len(current_players))
                if self.game_board.state.find('K') == -1:
                    winnner = "b"
                if self.game_board.state.find('k') == -1:
                    winnner = "w"
                z[np.array(current_players) == winnner] = 1.0
                z[np.array(current_players) != winnner] = -1.0
                game_over = True
                print("Game end. Winner is player : ", winnner, " In {} steps".format(self.game_board.round - 1))
            elif self.game_board.restrict_round >= 60:
                z = np.zeros(len(current_players))
                game_over = True
                print("Game end. Tie in {} steps".format(self.game_board.round - 1))
            # elif(self.mcts.root.v < self.resign_threshold):
            #     pass
            # elif(self.mcts.root.Q < self.resign_threshold):
            #    pass
            if game_over:
                # self.mcts.root = leaf_node(None, self.mcts.p_, "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
                self.mcts.reload()
        print("Using time {} s".format(time.time() - start_time))
        return zip(states, mcts_probs, z), len(z)

    def policy_update(self, batch_iter):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # print("training data_buffer len : ", len(self.data_buffer))
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        winner_batch = np.expand_dims(winner_batch, 1)

        start_time = time.time()
        old_probs, old_v = self.mcts.forward(state_batch)
        for i in range(self.epochs):
            # print("tf.executing_eagerly() : ", tf.executing_eagerly())
            state_batch = np.array(state_batch)
            if len(state_batch.shape) == 3:
                sp = state_batch.shape
                state_batch = np.reshape(state_batch, [1, sp[0], sp[1], sp[2]])
            self.policy_value_network.train(state_batch, batch_iter, i, lr=self.lr_multiplier * self.learning_rate)

            new_probs, new_v = self.mcts.forward(state_batch)
            kl_tmp = old_probs * (np.log((old_probs + 1e-10) / (new_probs + 1e-10)))

            kl_lst = []
            for line in kl_tmp:
                # print("line.shape", line.shape)
                all_value = [x for x in line if str(x) != 'nan' and str(x) != 'inf']  # 除去inf值
                kl_lst.append(np.sum(all_value))
            kl = np.mean(kl_lst)
            # kl = scipy.stats.entropy(old_probs, new_probs)
            # kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        self.policy_value_network.save_checkpoint()
        print("train using time {} s".format(time.time() - start_time))

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v) / np.var(
            np.array(winner_batch))  # .flatten()
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v) / np.var(
            np.array(winner_batch))  # .flatten()
        print(
            "kl:{:.5f},lr_multiplier:{:.3f},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, explained_var_old, explained_var_new))
        # return loss, accuracy

    def learn(self):
        # self.game_loop
        batch_iter = 0
        try:
            while True:
                batch_iter += 1
                play_data, episode_len = self.execute_episode()
                print("batch i:{}, episode_len:{}".format(batch_iter, episode_len))
                extend_data = []
                # states_data = []
                for state, mcts_prob, winner in play_data:
                    states_data = self.mcts.state_to_positions(state)
                    # prob = np.zeros(labels_len)
                    # for idx in range(len(mcts_prob[0][0])):
                    #     prob[label2i[mcts_prob[0][0][idx]]] = mcts_prob[0][1][idx]
                    extend_data.append((states_data, mcts_prob, winner))
                self.data_buffer.extend(extend_data)
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update(batch_iter)
                # if (batch_iter) % self.game_batch == 0:
                #     print("current self-play batch: {}".format(batch_iter))
                #     win_ratio = self.policy_evaluate()
        except KeyboardInterrupt:
            self.policy_value_network.save_checkpoint()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)


from china_chess.algorithm.sl_net import NNetWrapper as nn
from utils import *


def execute_episode(numIters, iter_number):
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                       pi is the MCTS informed policy vector, v is +1 if
                       the player eventually won the game, else -1.
    """
    nnet = nn()
    mcts = MCTS(policy_loop_arg=True, net=nnet)
    print(f"进程 {os.getpid()} 开启")
    gs = GameState()
    train_examples = []
    episode_step = 0

    peace_round = 0
    remain_piece = countpiece(gs.state_str)
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
        current_player = gs.get_current_player()
        gs.do_move(move)
        is_end, winner, info = gs.game_end()
        mcts.update_with_move(move)

        remain_piece_round = countpiece(gs.state_str)
        if remain_piece_round < remain_piece:
            remain_piece = remain_piece_round
            peace_round = 0
        else:
            peace_round += 1

        temp = [x.strip() for x in gs.display()]
        msg = str("\n".join(temp)) + "\n执行的行为是{}".format(move) + "\n执行该行为的玩家为{}".format(
            current_player) + "\n当前玩家为{}".format(gs.get_current_player())
        write_line(file_name="_execute_episode_procedure_" + str(numIters) + "_" + str(iter_number), msg=msg,
                   title="在execute_episode方法中的过程：" + info)

        if episode_step > 150 and peace_round > 60:
            for t in range(len(train_examples)):
                train_examples[t][2] = 0
            temp = [x.strip() for x in gs.display()]
            msg = str("\n".join(temp))
            write_line(file_name="_execute_episode_terminal_", msg=msg,
                       title="在execute_episode方法中的终结局面(和棋)：" + info)
            return train_examples

        if is_end:
            temp = [x.strip() for x in gs.display()]
            msg = str("\n".join(temp)) + "\n执行的行为是{}".format(move) + "\n执行该行为的玩家为{}".format(
                current_player) + "\n当前玩家为{}".format(gs.get_current_player())
            write_line(file_name="_execute_episode_terminal_", msg=msg,
                       title="在execute_episode方法中的终结局面：" + info)
            for t in range(len(train_examples)):
                if winner == gs.get_current_player():
                    train_examples[t][2] = 1
                else:
                    train_examples[t][2] = -1
            return train_examples


from concurrent.futures import ProcessPoolExecutor
import cloudpickle


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
