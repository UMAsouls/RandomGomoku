
from N_Tuple import NTupleNetwork
from Interfaces import IEnv

import numpy as np
import time

BATCH_SIZE = 32
BUFFER_SIZE = 10000
LEARNING_RATE = 0.01
GAMMA = 0.9

SEARCH_TIME = 0.05

CPUCT = 1.0

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
    
    def GetActions(self) -> np.ndarray:
        actions: np.ndarray = np.array(list(self._childs.keys()))
        
        return actions
        
        
    @property
    def is_leaf(self) -> bool:
        return len(self._childs) <= 0
    
        
        

class NTupleMCTSAgent:
    def __init__(self, env: IEnv, board_size=19, model_path = "NTupleMCTSModel", net_list:list[int] = [10]):
        self.policy_network = NTupleNetwork(board_size, net_list, LEARNING_RATE)
        self.value_network = NTupleNetwork(board_size, net_list, LEARNING_RATE)
        
        self.env = env
        
        self.board_size = board_size
        
        self.root = Node(None, 1.0, board_size**2)
        
    def Search(self) -> int:
        
        dat = self.env.backup()
        start_time = time.time()
        
        while time.time() - start_time < SEARCH_TIME:
            self.__search_once()
            self.env.restore(dat)
            
        
        N_values = self.root.GetChildrenNs()
        actions = self.root.GetActions()
        max_n = np.max(N_values)
        indices = np.where(N_values == max_n)[0]
        
        action = actions[np.random.choice(indices)]
        
        return action
            
        
        
    def __search_once(self) -> int:
        node = self.root
        done = False
        
        while not node.is_leaf:
            act, node = node.select(CPUCT)
            
            (x,y) = (act%self.board_size, act//self.board_size)
            _, reward, done, _ = self.env.step((x,y))
            
        if not done:
            value = self.__expand_func(node)
        else:
            value = reward
            
        while node is not None:
            node.update(value=value)
            node = node.Parent
            value = -value
            
            
    def __expand_func(self, node: Node) -> float:
        board: np.ndarray = self.env.GetBoard_CurrentPlayer()
        
        legal_move = self.env.GetLegalAction()
        p_scores = self.policy_network.evaluate(board)
        
        probs = self.__softmax(p_scores, legal_move)
        
        node.expand(legal_move,probs)
        
        value = np.max(self.value_network.evaluate(board))
        
        return value       
        
    def __softmax(self, scores: np.ndarray, legal_move: np.ndarray) -> np.ndarray:
        #合法手に対するスコアを確率に変換
        scores = scores[legal_move] #合法手のみにする
        scores = np.exp(scores - np.max(scores))
        probs = scores / np.sum(scores)
        
        return probs
        
        
        
            
        
        
        
            
            
    