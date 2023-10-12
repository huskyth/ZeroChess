import asyncio
from asyncio import Queue
from collections import defaultdict

import numpy as np

from china_chess.algorithm.cchess.const_function import c_PUCT, ind, is_kill_move, virtual_loss, flip_policy
from china_chess.algorithm.cchess.game_board import GameBoard
from china_chess.algorithm.cchess.leaf_node import leaf_node
from threading import Lock

from china_chess.algorithm.mcts_async import QueueItem


class MCTS_tree(object):
    def __init__(self, in_state, in_forward, search_threads):
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3  # 0.03
        self.p_ = (1 - self.noise_eps) * 1 + self.noise_eps * np.random.dirichlet([self.dirichlet_alpha])
        self.root = leaf_node(None, self.p_, in_state)
        self.c_puct = 5  # 1.5
        # self.policy_network = in_policy_network
        self.forward = in_forward
        self.node_lock = defaultdict(Lock)

        self.virtual_loss = 3
        self.now_expanding = set()
        self.expanded = set()
        self.cut_off_depth = 30
        # self.QueueItem = namedtuple("QueueItem", "feature future")
        self.sem = asyncio.Semaphore(search_threads)
        self.queue = Queue(search_threads)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

    def reload(self):
        self.root = leaf_node(None, self.p_,
                              "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")  # "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
        self.expanded = set()

    def Q(self, move) -> float:
        ret = 0.0
        find = False
        for a, n in self.root.child.items():
            if move == a:
                ret = n.Q
                find = True
        if (find == False):
            print("{} not exist in the child".format(move))
        return ret

    def update_tree(self, act):
        # if(act in self.root.child):
        self.expanded.discard(self.root)
        self.root = self.root.child[act]
        self.root.parent = None
        # else:
        #     self.root = leaf_node(None, self.p_, in_state)

    # def do_simulation(self, state, current_player, restrict_round):
    #     node = self.root
    #     last_state = state
    #     while(node.is_leaf() == False):
    #         # print("do_simulation while current_player : ", current_player)
    #         with self.node_lock[node]:
    #             action, node = node.select(self.c_puct)
    #         current_player = "w" if current_player == "b" else "b"
    #         if is_kill_move(last_state, node.state) == 0:
    #             restrict_round += 1
    #         else:
    #             restrict_round = 0
    #         last_state = node.state
    #
    #     positions = self.generate_inputs(node.state, current_player)
    #     positions = np.expand_dims(positions, 0)
    #     action_probs, value = self.forward(positions)
    #     if self.is_black_turn(current_player):
    #         action_probs = cchess_main.flip_policy(action_probs)
    #
    #     # print("action_probs shape : ", action_probs.shape)    #(1, 2086)
    #     with self.node_lock[node]:
    #         if(node.state.find('K') == -1 or node.state.find('k') == -1):
    #             if (node.state.find('K') == -1):
    #                 value = 1.0 if current_player == "b" else -1.0
    #             if (node.state.find('k') == -1):
    #                 value = -1.0 if current_player == "b" else 1.0
    #         elif restrict_round >= 60:
    #             value = 0.0
    #         else:
    #             moves = GameBoard.get_legal_moves(node.state, current_player)
    #             # print("current_player : ", current_player)
    #             # print(moves)
    #             node.expand(moves, action_probs)
    #
    #         # if(node.parent != None):
    #         #     node.parent.N += self.virtual_loss
    #         node.N += self.virtual_loss
    #         node.W += -self.virtual_loss
    #         node.Q = node.W / node.N
    #
    #     # time.sleep(0.1)
    #
    #     with self.node_lock[node]:
    #         # if(node.parent != None):
    #         #     node.parent.N += -self.virtual_loss# + 1
    #         node.N += -self.virtual_loss# + 1
    #         node.W += self.virtual_loss# + leaf_v
    #         # node.Q = node.W / node.N
    #
    #         node.backup(-value)

    def is_expanded(self, key) -> bool:
        """Check expanded status"""
        return key in self.expanded

    async def tree_search(self, node, current_player, restrict_round) -> float:
        """Independent MCTS, stands for one simulation"""
        self.running_simulation_num += 1

        # reduce parallel search number
        with await self.sem:
            value = await self.start_tree_search(node, current_player, restrict_round)
            # logger.debug(f"value: {value}")
            # logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')
            self.running_simulation_num -= 1

            return value

    async def start_tree_search(self, node, current_player, restrict_round) -> float:
        """Monte Carlo Tree search Select,Expand,Evauate,Backup"""
        now_expanding = self.now_expanding

        while node in now_expanding:
            await asyncio.sleep(1e-4)

        if not self.is_expanded(node):  # and node.is_leaf()
            """is leaf node try evaluate and expand"""
            # add leaf node to expanding list
            self.now_expanding.add(node)

            positions = self.generate_inputs(node.state, current_player)
            # positions = np.expand_dims(positions, 0)

            # push extracted dihedral features of leaf node to the evaluation queue
            future = await self.push_queue(positions)  # type: Future
            await future
            action_probs, value = future.result()

            # action_probs, value = self.forward(positions)
            if self.is_black_turn(current_player):
                action_probs = flip_policy(action_probs)

            moves = GameBoard.get_legal_moves(node.state, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)
            self.expanded.add(node)  # node.state

            # remove leaf node from expanding list
            self.now_expanding.remove(node)

            # must invert, because alternative layer has opposite objective
            return value[0] * -1

        else:
            """node has already expanded. Enter select phase."""
            # select child node with maximum action scroe
            last_state = node.state

            action, node = node.select_new(c_PUCT)
            current_player = "w" if current_player == "b" else "b"
            if is_kill_move(last_state, node.state) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_state = node.state

            # action_t = self.select_move_by_action_score(key, noise=True)

            # add virtual loss
            # self.virtual_loss_do(key, action_t)
            node.N += virtual_loss
            node.W += -virtual_loss

            # evolve game board status
            # child_position = self.env_action(position, action_t)

            if (node.state.find('K') == -1 or node.state.find('k') == -1):
                if (node.state.find('K') == -1):
                    value = 1.0 if current_player == "b" else -1.0
                if (node.state.find('k') == -1):
                    value = -1.0 if current_player == "b" else 1.0
                value = value * -1
            elif restrict_round >= 60:
                value = 0.0
            else:
                value = await self.start_tree_search(node, current_player, restrict_round)  # next move
            # if node is not None:
            #     value = await self.start_tree_search(node)  # next move
            # else:
            #     # None position means illegal move
            #     value = -1

            # self.virtual_loss_undo(key, action_t)
            node.N += -virtual_loss
            node.W += virtual_loss

            # on returning search path
            # update: N, W, Q, U
            # self.back_up_value(key, action_t, value)
            node.back_up_value(value)  # -value

            # must invert
            return value * -1
            # if child_position is not None:
            #     return value * -1
            # else:
            #     # illegal move doesn't mean much for the opponent
            #     return 0

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            # logger.debug(f"predicting {len(item_list)} items")
            features = np.asarray([item.feature for item in item_list])  # asarray
            # print("prediction_worker [features.shape] before : ", features.shape)
            # shape = features.shape
            # features = features.reshape((shape[0] * shape[1], shape[2], shape[3], shape[4]))
            # print("prediction_worker [features.shape] after : ", features.shape)
            # policy_ary, value_ary = self.run_many(features)
            action_probs, value = self.forward(features)
            for p, v, item in zip(action_probs, value, item_list):
                item.future.set_result((p, v))

    async def push_queue(self, features):
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    # @profile
    def main(self, state, current_player, restrict_round, playouts):
        node = self.root
        if not self.is_expanded(node):  # and node.is_leaf()    # node.state
            # print('Expadning Root Node...')
            positions = self.generate_inputs(node.state, current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.forward(positions)
            if self.is_black_turn(current_player):
                action_probs = flip_policy(action_probs)

            moves = GameBoard.get_legal_moves(node.state, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)
            self.expanded.add(node)  # node.state

        coroutine_list = []
        for _ in range(playouts):
            coroutine_list.append(self.tree_search(node, current_player, restrict_round))
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

    def do_simulation(self, state, current_player, restrict_round):
        node = self.root
        last_state = state
        while (node.is_leaf() == False):
            # print("do_simulation while current_player : ", current_player)
            action, node = node.select(self.c_puct)
            current_player = "w" if current_player == "b" else "b"
            if is_kill_move(last_state, node.state) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_state = node.state

        positions = self.generate_inputs(node.state, current_player)
        positions = np.expand_dims(positions, 0)
        action_probs, value = self.forward(positions)
        if self.is_black_turn(current_player):
            action_probs = flip_policy(action_probs)

        # print("action_probs shape : ", action_probs.shape)    #(1, 2086)

        if (node.state.find('K') == -1 or node.state.find('k') == -1):
            if (node.state.find('K') == -1):
                value = 1.0 if current_player == "b" else -1.0
            if (node.state.find('k') == -1):
                value = -1.0 if current_player == "b" else 1.0
        elif restrict_round >= 60:
            value = 0.0
        else:
            moves = GameBoard.get_legal_moves(node.state, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)

        node.backup(-value)

    def generate_inputs(self, in_state, current_player):
        state, palyer = self.try_flip(in_state, current_player, self.is_black_turn(current_player))
        return self.state_to_positions(state)

    def replace_board_tags(self, board):
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.replace("/", "")

    # 感觉位置有点反了，当前角色的棋子在右侧，plane的后面
    def state_to_positions(self, state):
        # TODO C plain x 2
        board_state = self.replace_board_tags(state)
        pieces_plane = np.zeros(shape=(9, 10, 14), dtype=np.float32)
        for rank in range(9):  # 横线
            for file in range(10):  # 直线
                v = board_state[rank * 9 + file]
                if v.isalpha():
                    pieces_plane[rank][file][ind[v]] = 1
        assert pieces_plane.shape == (9, 10, 14)
        return pieces_plane

    def try_flip(self, state, current_player, flip=False):
        if not flip:
            return state, current_player

        rows = state.split('/')

        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a

        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(row) for row in reversed(rows)]), ('w' if current_player == 'b' else 'b')

    def is_black_turn(self, current_player):
        return current_player == 'b'
