import time

import numpy as np
import copy
from china_chess.algorithm.icy_chess.chess_board_from_icy import BaseChessBoard
from china_chess.algorithm.icy_chess.common_board import flipped_uci_labels, create_uci_labels
from china_chess.algorithm.icy_chess.game_board import GameBoard
from china_chess.algorithm.icy_chess.game_convert import boardarr2netinput
from china_chess.algorithm.icy_chess.game_state import GameState

uci_labels = create_uci_labels()


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, state, noise=False):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.virtual_loss = 0
        self.noise = noise
        self.state = state

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # dirichlet noise should be applied when every select action
        if False and self.noise == True and self._parent == None:
            noise_d = np.random.dirichlet([0.3] * len(action_priors))
            for (action, prob), one_noise in zip(action_priors, noise_d):
                if action not in self._children:
                    prob = (1 - 0.25) * prob + 0.25 * one_noise
                    next_state = copy.deepcopy(self.state)
                    next_state.do_move(action)
                    self._children[action] = TreeNode(self, prob, next_state, noise=self.noise)
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    next_state = copy.deepcopy(self.state)
                    next_state.do_move(action)
                    self._children[action] = TreeNode(self, prob, next_state)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        if not self.noise:
            max_move = max(self._children.items(),
                           key=lambda act_node: act_node[1].get_value(c_puct))
            state_temp = copy.deepcopy(self.state)
            state_temp.do_move(max_move[0])
            if state_temp.game_end()[0] is True:
                pass
            return max_move
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
            # print("Q={},U={},VL={}".format(self._Q, self._u, self.virtual_loss))
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

    def __init__(self, c_puct=5, n_playout=1200, virtual_loss=3,
                 dnoise=False, name="MCTS", remote=None):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0, GameState(), noise=dnoise)
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.virtual_loss = virtual_loss

        self.select_time = 0
        self.policy_time = 0
        self.update_time = 0

        self.dnoise = dnoise

        self.name = name

        self.remote = remote

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        if True:
            node = self._root
            road = []
            while True:
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

                node.virtual_loss = - np.inf
                # node.update_recursive(leaf_value)
                self.update_time += (time.time() - start)
                return

            start = time.time()

            action_probs, leaf_value = self._policy(state)
            self.policy_time += (time.time() - start)

            start = time.time()
            end, winner, info = state.game_end()
            if not end:
                node.expand(action_probs)
            else:
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
            self.update_time += (time.time() - start)

    def get_move_probs(self, state, temp=0, can_apply_dnoise=False):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        if not can_apply_dnoise:
            self._root.noise = False
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        if temp == 0:
            bestAs = np.array(np.argwhere(visits == np.max(visits))).flatten()
            bestA = np.random.choice(bestAs)
            return acts[bestA]

        counts = [x ** (1. / temp) for x in visits]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        aMove = np.random.choice(len(probs), p=probs)

        return acts[aMove]

    def update_with_move(self, last_move, allow_legacy=True):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        self.num_proceed = 0
        if last_move in self._root._children and allow_legacy:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            init = copy.deepcopy(self._root.state)
            if last_move != -1:
                init.do_move(last_move)
            else:
                init = GameState()
            self._root = TreeNode(None, 1.0, init, noise=self.dnoise)

    def __str__(self):
        return "MCTS" + " " + self.name

    def _policy(self, state):
        bb = BaseChessBoard(state.state_str)
        state_str = bb.get_board_arr()
        net_x = boardarr2netinput(state_str, state.get_current_player())

        self.remote.send(('predict', net_x))
        policy_out, val_out = self.remote.recv()




        legal_move = GameBoard.get_legal_moves(state.state_str, state.get_current_player())
        legal_move = set(legal_move)
        legal_move_b = set(flipped_uci_labels(legal_move))

        action_probs = []
        if state.current_player == 'b':
            for move, prob in zip(uci_labels, policy_out[0]):
                if move in legal_move_b:
                    move = flipped_uci_labels([move])[0]
                    action_probs.append((move, prob))
        else:
            for move, prob in zip(uci_labels, policy_out[0]):
                if move in legal_move:
                    action_probs.append((move, prob))
        return action_probs, val_out
