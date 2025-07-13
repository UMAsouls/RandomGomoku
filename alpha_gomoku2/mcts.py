import numpy as np
import copy
import torch
from GomokuEnv import GomokuEnv
from RandomGomoku.Board import Board
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # アクションからTreeNodeへのマップ
        self._n_visits = 0  # このノードの訪問回数
        self._Q = 0  # このノードの価値の平均（Q値）
        self._u = 0  # UCT項（Upper Confidence Bound）
        self._P = prior_p  # 事前確率（ニューラルネットワークからの出力）
    
    def expand(self, action_priors):
        """ツリーを拡張して新しい子ノードを作成する。
        action_priors: アクションとそのポリシー関数による事前確率の
            タプルのリスト。
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
    
    def select(self, c_puct):
        """子ノードの中から最大のアクション価値Q + ボーナスu(P)を与える
        アクションを選択する。
        c_puct: 探索のバランスを制御するパラメータ
        Return: (action, next_node)のタプル
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
        
    def update(self, leaf_value):
        """リーフ評価からノード価値を更新する。
        leaf_value: 現在のプレイヤーの視点からのサブツリー評価値
        """
        # 訪問回数をカウント
        self._n_visits += 1
        # Q値を更新（全訪問での価値の移動平均）
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        
    def update_recursive(self, leaf_value):
        """update()の再帰版。全ての祖先ノードに対して適用される。
        MCTSでは、リーフから根までの経路上の全ノードを更新する必要がある。
        """
        # 根ノードでなければ、親ノードを先に更新する
        if self._parent:
            # 親ノードの視点では価値が反転するため-leaf_valueを渡す
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        """このノードの価値を計算して返す。
        リーフ評価Q値と、訪問回数で調整された事前確率uの組み合わせ。
        これがUCT（Upper Confidence Bound applied to Trees）の核心部分。
        c_puct: 価値Qと事前確率Pの相対的影響を制御する数値 (0, inf)
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """リーフノードかどうかを確認（拡張されていないノード）。"""
        return self._children == {}

    def is_root(self):
        """根ノードかどうかを確認。"""
        return self._parent is None
def action_to_location(action, board_size):
    """アクションを2次元の座標に変換する。
    action: 1次元のインデックス（0からboard_size*board_size-1まで）
    board_size: 盤面のサイズ
    Return: (x, y)のタプル
    """
    y = action // board_size
    x = action % board_size
    return x, y
class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
    def _playout(self, env:GomokuEnv):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        done = False
        reward = 0
        while(True):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            
            _, reward, done, _ = env.step(action_to_location(action, env.board_size))
        
        # Check if game is finished after reaching leaf
        if not done:
            # Game is not finished, get policy evaluation
            action_priors, leaf_value = self._policy(env.board)
            node.expand(action_priors)
        else:
            # Game is finished, determine leaf value based on reward
            if reward == -1:
                leaf_value = 0
            else:
                #TODO:等式が意味あってるか確認
                leaf_value = 1 if reward == env.train_player else -1
        
        # If game is not finished, we still need to get leaf_value
        if not done:
            action_priors, leaf_value = self._policy(env.board)
        
        node.update_recursive(-leaf_value)
        
    def get_move_probs(self, env:GomokuEnv, temp=1e-3):
        """Run MCTS to get move probabilities.
        env: the Gomoku environment to play in.
        Return: a list of (action, probability) tuples.
        """
        for _ in range(self._n_playout):
            # Copy the environment to avoid modifying the original state.
            env_copy = copy.deepcopy(env)
            self._playout(env_copy)
        
        # Get the visit counts for each action from the root node.
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        
        return acts, act_probs
    
    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)
    def get_legal_positions(self,board_size, board: Board):
        #形式: 整数のリスト（例：[0, 1, 2, 5, 7, 10, ...]）
        # 意味: 各整数は盤面上の空いているマス目の位置を1次元のインデックスで表現
        # 範囲: 0 ～ (board_size × board_size - 1)
        state = board.GetBoardInt()
        legal_positions = []
        for y in range(board_size):
            for x in range(board_size):
                if state[y][x] == 0:
                    legal_positions.append(x+ y * board_size)
        # print(f"有効な手の数: {len(legal_positions)}")
        # print(f"有効な手の位置: {legal_positions}")
        return legal_positions
    def get_action(self, env:GomokuEnv, temp=1e-3, return_prob=0):
        
        sensible_moves = self.get_legal_positions(env.board_size,env.board)
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(env.board_size * env.board_size)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 探索のためにディリクレノイズを追加（自己対戦訓練に必要）
                # ディリクレノイズは探索の多様性を保つため、未知の手も試すようになる
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 根ノードを更新し、探索ツリーを再利用
                # 次の手番でも部分的な探索結果を活用できる
                self.mcts.update_with_move(move)
            else:
                # デフォルトのtemp=1e-3では、最も確率の高い手を選ぶのとほぼ同じ
                # 温度パラメータが低いため、ほぼ決定論的な選択になる
                move = np.random.choice(acts, p=probs)
                # 根ノードをリセット
                # 対戦相手の手が予測できないため、ツリーを再構築
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            # アクションを2次元の座標に変換
            move = action_to_location(move, env.board_size)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)