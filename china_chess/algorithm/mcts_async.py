import asyncio
from asyncio.queues import Queue
import time
from collections import namedtuple

import numpy as np
import copy

from china_chess.algorithm.icy_chess.chess_board_from_icy import BaseChessBoard
from china_chess.algorithm.icy_chess.common_board import flipped_uci_labels, create_uci_labels
from china_chess.algorithm.icy_chess.game_board import GameBoard
from china_chess.algorithm.icy_chess.game_convert import boardarr2netinput
from china_chess.constant import SL_MODEL_PATH
from othello.pytorch.NNet import NNetWrapper

queue = Queue(400)
QueueItem = namedtuple("QueueItem", "feature future")
uci_labels = create_uci_labels()


async def push_queue(features, loop):
    future = loop.create_future()
    item = QueueItem(features, future)
    await queue.put(item)
    return future


async def policy_value_fn_queue_of_my_net(state, loop):
    bb = BaseChessBoard(state.state_str)
    state_str = bb.get_board_arr()
    net_x = boardarr2netinput(state_str, state.get_current_player())
    future = await push_queue(net_x, loop)
    await future
    policy_out, val_out = future.result()
    legal_move = GameBoard.get_legal_moves(state.state_str, state.get_current_player())
    legal_move = set(legal_move)
    legal_move_b = set(flipped_uci_labels(legal_move))

    action_probs = []
    if state.current_player == 'b':
        for move, prob in zip(uci_labels, policy_out):
            if move in legal_move_b:
                move = flipped_uci_labels([move])[0]
                action_probs.append((move, prob))
    else:
        for move, prob in zip(uci_labels, policy_out):
            if move in legal_move:
                action_probs.append((move, prob))
    return action_probs, val_out


async def policy_value_fn_queue(state, loop):
    bb = BaseChessBoard(state.state_str)
    state_str = bb.get_board_arr()
    net_x = np.transpose(boardarr2netinput(state_str, state.get_current_player()), [1, 2, 0])
    net_x = np.expand_dims(net_x, 0)
    future = await push_queue(net_x, loop)
    await future
    policy_out, val_out = future.result()
    policy_out = [1] * 2086
    # policyout,valout = sess.run([net_softmax,value_head],feed_dict={X:net_x,training:False})
    # result = work.delay((state.statestr,state.get_current_player()))
    # while True:
    #    if result.ready():
    #        policyout,valout = result.get()
    #        break
    #    else:
    #        await asyncio.sleep(1e-3)
    # policyout,valout = policyout[0],valout[0][0]
    # policyout, valout = policyout, valout[0]
    legal_move = GameBoard.get_legal_moves(state.state_str, state.get_current_player())
    # if state.currentplayer == 'b':
    #    legal_move = board.flipped_uci_labels(legal_move)
    legal_move = set(legal_move)
    legal_move_b = set(flipped_uci_labels(legal_move))

    action_probs = []
    if state.current_player == 'b':
        for move, prob in zip(uci_labels, policy_out):
            if move in legal_move_b:
                move = flipped_uci_labels([move])[0]
                action_probs.append((move, prob))
    else:
        for move, prob in zip(uci_labels, policy_out):
            if move in legal_move:
                action_probs.append((move, prob))
    # action_probs = sorted(action_probs,key=lambda x:x[1])
    return action_probs, val_out


async def prediction_worker(mcts_policy_async):
    q = queue
    while mcts_policy_async.num_proceed < mcts_policy_async._n_playout:
        if q.empty():
            await asyncio.sleep(1e-3)
            continue
        item_list = [q.get_nowait() for _ in range(q.qsize())]
        # print("processing : {} samples".format(len(item_list)))
        features = np.concatenate([np.expand_dims(item.feature, axis=0) for item in item_list], axis=0)

        action_probs, value = mcts_policy_async.net.predict(features)
        for p, v, item in zip(action_probs, value, item_list):
            item.future.set_result((p, v))


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, noise=False):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.virtual_loss = 0
        self.noise = noise

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # dirichlet noise should be applied when every select action
        if False and self.noise == True and self._parent == None:
            # print("noise")
            noise_d = np.random.dirichlet([0.3] * len(action_priors))
            for (action, prob), one_noise in zip(action_priors, noise_d):
                if action not in self._children:
                    prob = (1 - 0.25) * prob + 0.25 * one_noise
                    self._children[action] = TreeNode(self, prob, noise=self.noise)
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        if not self.noise:
            return max(self._children.items(),
                       key=lambda act_node: act_node[1].get_value(c_puct))
        elif self.noise and self._parent is not None:
            return max(self._children.items(),
                       key=lambda act_node: act_node[1].get_value(c_puct))
        else:
            noise_d = np.random.dirichlet([0.3] * len(self._children))
            return max(list(zip(noise_d, self._children.items())),
                       key=lambda act_node: act_node[1][1].get_value(c_puct, noise_p=act_node[0]))[1]

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct, noise_p=None):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        if noise_p is None:
            self._u = (c_puct * self._P *
                       np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
            return self._Q + self._u + self.virtual_loss
        else:
            self._u = (c_puct * (self._P * 0.75 + noise_p * 0.25) *
                       np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
            return self._Q + self._u + self.virtual_loss

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, search_threads=32, virtual_loss=3,
                 policy_loop_arg=False, dnoise=False, ):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0, noise=dnoise)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.virtual_loss = virtual_loss
        self.loop = asyncio.get_event_loop()
        self.policy_loop_arg = policy_loop_arg
        self.sem = asyncio.Semaphore(search_threads)
        self.now_expanding = set()

        self.select_time = 0
        self.policy_time = 0
        self.update_time = 0

        self.num_proceed = 0
        self.dnoise = dnoise

        self.net = NNetWrapper()
        self.net.load_checkpoint(folder=SL_MODEL_PATH, filename="best_loss.pth.tar")

    async def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        async with self.sem:
            node = self._root
            road = []
            while True:
                while node in self.now_expanding:
                    await asyncio.sleep(1e-4)
                start = time.time()
                if node.is_leaf():
                    break
                # Greedily select next move.
                action, node = node.select(self._c_puct)
                road.append(node)
                node.virtual_loss -= self.virtual_loss
                state.do_move(action)
                self.select_time += (time.time() - start)

            # at leave node if long check or long catch then cut off the node
            if state.should_cutoff():
                # cut off node
                for one_node in road:
                    one_node.virtual_loss += self.virtual_loss
                # now at this time, we do not update the entire tree branch, the accuracy loss is supposed to be small
                # node.update_recursive(-leaf_value)

                # set virtual loss to -inf so that other threads would not visit the same node again(so the node is cut off)
                node.virtual_loss = - np.inf
                # node.update_recursive(leaf_value)
                self.update_time += (time.time() - start)
                # however the proceed number still goes up 1
                self.num_proceed += 1
                return

            start = time.time()
            self.now_expanding.add(node)
            # Evaluate the leaf using a network which outputs a list of
            # (action, probability) tuples p and also a score v in [-1, 1]
            # for the current player
            if not self.policy_loop_arg:
                action_probs, leaf_value = await self._policy(state)
            else:
                action_probs, leaf_value = await self._policy(state, self.loop)
            self.policy_time += (time.time() - start)

            start = time.time()
            # Check for end of game.
            end, winner = state.game_end()
            if not end:
                node.expand(action_probs)
            else:
                # for end state，return the "true" leaf_value
                if winner == -1:  # tie
                    leaf_value = 0.0
                else:
                    leaf_value = (
                        1.0 if winner == state.get_current_player() else -1.0
                    )

            # Update value and visit count of nodes in this traversal.
            for one_node in road:
                one_node.virtual_loss += self.virtual_loss
            node.update_recursive(-leaf_value)
            self.now_expanding.remove(node)
            # node.update_recursive(leaf_value)
            self.update_time += (time.time() - start)
            self.num_proceed += 1

    def get_move_probs(self, state, temp=1e-3, verbose=False, predict_workers=[], can_apply_dnoise=False):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        if not can_apply_dnoise:
            self._root.noise = False
        coroutine_list = []
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            coroutine_list.append(self._playout(state_copy))
        coroutine_list += predict_workers
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move, allow_legacy=True):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        self.num_proceed = 0
        if last_move in self._root._children and allow_legacy:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, noise=self.dnoise)

    def __str__(self):
        return "MCTS"
