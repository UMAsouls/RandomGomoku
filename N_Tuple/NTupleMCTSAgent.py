
from N_Tuple import NTupleNetwork

from N_Tuple.Debug import MCTSMemo
from Interfaces import IEnv

import numpy as np
import time

LEARNING_RATE = 0.01
GAMMA = 0.9

SEARCH_TIME = 0.05

CPUCT = 1.0

POLICY_NET_PATH = "PolicyNet"
VALUE_NET_PATH = "ValueNet"

class Node:
    def __init__(self, parent: "Node", prior_p: float, node_max: int):
        #訪問回数
        self._N = 0
        #勝利回数
        self._W = 0
        
        #事前確率
        self._P = prior_p
        
        #親ノード
        self._parent: "Node" = parent
        #子ノード
        self._childs: dict[int,"Node"] = {}
        
        #ノードの限界
        self.node_max = node_max
        
        
    @property
    def N(self) -> int:
        return self._N
    
    @property
    def W(self) -> int:
        return self._W
    
    @property
    def Q(self) -> float:
        if(self._N == 0): return 0
        else: return self._W/self._N
    
    @property
    def P(self) -> float:
        return self._P
    
    @property
    def Parent(self) -> "Node":
        return self._parent
    
    
    def select(self, c_puct: float) -> tuple[int,"Node"]:
        actions: list[int]
        children: list["Node"]
        actions, children = zip(*self._childs.items())
        
        N_values = np.array([i.N for i in children], dtype=np.int64)
        Q_values = np.array([i.Q for i in children], dtype=np.float64)
        P_values = np.array([i.P for i in children], dtype=np.float64)
        
        U_values: np.ndarray = c_puct *  P_values * (np.sqrt(self._N)/(1+N_values))
        
        puct_scores = Q_values + U_values
        
        max_score = np.max(puct_scores)
        max_indices = np.where(puct_scores == max_score)[0]
        best_index = np.random.choice(max_indices)

        best_action = actions[best_index]
        best_node = children[best_index]
        
        return best_action, best_node
    
    def expand(self, acts: list[int], probs: list[float]) -> None:
        nodes = [Node(self, p, self.node_max) for p in probs]
        
        self._childs = dict(zip(acts,nodes))
        
    def update(self, value: float) -> None:
        self._N += 1
        self._W += value
        
    def GetChildrenNs(self) -> np.ndarray:
        children: list["Node"] = list(self._childs.values())
        
        return np.array([i.N for i in children], dtype=np.int64)
    
    def GetLegalActions(self) -> np.ndarray:
        actions: np.ndarray = np.array(list(self._childs.keys()))
        
        return actions
        
        
    @property
    def is_leaf(self) -> bool:
        return len(self._childs) <= 0
    
        
        

class NTupleMCTSAgent:
    def __init__(
        self, env: IEnv, board_size=19, model_path = "NTupleMCTSModel", net_list:list[int] = [10],
        cpuct:float = CPUCT, search_time: float = SEARCH_TIME
        ):
        self.policy_network = NTupleNetwork(board_size, net_list, LEARNING_RATE)
        self.value_network = NTupleNetwork(board_size, net_list, LEARNING_RATE)
        
        self.env = env
        
        self.board_size = board_size
        
        self.root = Node(None, 1.0, board_size**2)
        
        self.policy_net_path = model_path + "/" + POLICY_NET_PATH
        self.value_net_path = model_path + "/" + VALUE_NET_PATH
        
        self.cpuct = cpuct
        self.search_time = search_time
        
        self.memo = MCTSMemo()
        
    def PrintMemo(self) -> None:
        self.memo.PrintMemo()
        
    def ResetMemo(self) -> None:
        self.memo.reset()
        
    def Search(self) -> tuple[tuple[int,int], np.ndarray]:
        self.memo.SearchStart()
        
        dat = self.env.backup()
        start_time = time.time()
        
        self.root = Node(None, 1.0, self.board_size**2)
        
        while time.time() - start_time < self.search_time:
            self.__search_once()
            self.env.restore(dat)
            
        
        N_values = self.root.GetChildrenNs()
        actions = self.root.GetLegalActions()
        max_n = np.max(N_values)
        indices = np.where(N_values == max_n)[0]
        
        action = actions[np.random.choice(indices)]
        
        probs = N_values/self.root.N
        indices = np.arange(actions.size)
        
        policy_target = np.ones(self.board_size**2, np.float64)*-1
        policy_target[actions[indices]] = probs[indices]
        
        self.memo.SearchEnd()
        return action, policy_target    
        
    def __search_once(self) -> int:
        node = self.root
        done = False
        
        while not node.is_leaf:
            self.memo.SelectStart()
            act, node = node.select(self.cpuct)
            
            (x,y) = (act%self.board_size, act//self.board_size)
            _, reward, done, _ = self.env.step((x,y))
            self.memo.SelectEnd()
            
        if not done:
            self.memo.ExpandStart()
            value = self.__expand_func(node)
            self.memo.ExpandEnd()
        else:
            value = reward
        
        depth = -1
        while node is not None:
            node.update(value=value)
            node = node.Parent
            value = -value
            depth += 1
            
        self.memo.depth_max_set(depth)
            
            
    def __expand_func(self, node: Node) -> float:
        board: np.ndarray = self.env.GetBoard_CurrentPlayer()
        
        legal_move = self.env.GetLegalAction()
        p_scores = self.policy_network.evaluate(board)
        
        probs = self.__softmax(p_scores[legal_move])
        
        node.expand(legal_move,probs)
        
        value = np.max(self.value_network.evaluate(board))
        
        return value       
        
    def __softmax(self, scores: np.ndarray) -> np.ndarray:
        #スコアを確率に変換
        scores = np.exp(scores - np.max(scores))
        probs = scores / np.sum(scores)
        
        return probs
    
    def train(self, state: np.ndarray, action: int, policies: np.ndarray, value: float) -> None:
        self.policy_train(state, policies)
        self.value_train(state, action, value)
    
    #方策の学習
    def policy_train(self, state: np.ndarray, target: np.ndarray) -> None:
        # ネットワークの現在の予測スコア(tanh後の値)を取得
        predicted_scores = self.policy_network.evaluate(state)
    
        # targetが-1でない箇所が合法手
        legal_actions = np.where(target >= 0)[0]
    
        # 合法手に対する予測スコアを抽出し、softmaxで確率に変換
        predicted_policy_probs = self.__softmax(predicted_scores[legal_actions])
    
        # 正解の方策(MCTSの訪問回数分布)も合法手のみを抽出
        target_policy_probs = target[legal_actions]
    
        # 予測と正解の誤差を計算 (交差エントロピー誤差の勾配)
        errors = predicted_policy_probs - target_policy_probs
    
        # 各合法手について、それぞれの誤差で更新
        # enumerateを使って、誤差配列のインデックス(i)とアクションID(act)を両方取得
        for i,act in enumerate(legal_actions):
            y = predicted_scores[act]
            error = errors[i]
            
            self.policy_network.learn(state, act, error, y)
    
    #価値の学習        
    def value_train(self, state: np.ndarray, action:int, target: float) -> None:
        predict_value = np.max(self.value_network.evaluate(state))
        error = target - predict_value
        
        self.value_network.learn(state, action, error, predict_value)
        
    def save(self) -> None:
        self.policy_network.save(self.policy_net_path)
        self.value_network.save(self.value_net_path)
        
    def load(self) -> None:
        self.policy_network.load(self.policy_net_path)
        self.value_network.load(self.value_net_path)
        
        
        
            
        
        
        
            
            
    