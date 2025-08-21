
from N_Tuple import NTupleNetwork, NTuplePVNetwork

from N_Tuple.Debug import MCTSMemo
from Interfaces import IEnv

import numpy as np
import time

LEARNING_RATE = 0.01
GAMMA = 0.9

SIMULATION_TIME = 100

CPUCT = 1.0

POLICY_NET_PATH = "PolicyNet"
VALUE_NET_PATH = "ValueNet"
PV_NET_PATH = "PVNet"

PARAMETER_PATH = "PARAMETER"

#ディリクレノイズ作成
EPSILON = 0.25
ALPHA = 0.3

class Node:
    def __init__(self, parent: "Node", prior_p: float, node_max: int, p_idx: int):
        #訪問回数
        self._N = 0
        #勝利回数
        self._W = 0
        
        #事前確率
        self._P = prior_p
        
        #親ノード
        self._parent: "Node" = parent
        self.p_idx = p_idx
        #子ノード
        self._childs: dict[int,"Node"] = {}
        
        #ノードの限界
        self.node_max = node_max
        
        self.N_values: np.ndarray
        self.Q_values: np.ndarray
        self.P_values: np.ndarray
        
        
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
    
    def update_child(self, N: int, Q: float, idx: int):
        self.N_values[idx] = N
        self.Q_values[idx] = Q
        
    
    def select(self, c_puct: float) -> tuple[int,"Node"]:
        actions: list[int]
        children: list["Node"]
        actions, children = zip(*self._childs.items())
        
        N_values = self.N_values
        Q_values = self.Q_values
        P_values = self.P_values
        
        U_values: np.ndarray = c_puct *  P_values * (np.sqrt(self._N)/(1+N_values))
        
        puct_scores = Q_values + U_values
        
        max_score = np.max(puct_scores)
        max_indices = np.where(puct_scores == max_score)[0]
        best_index = np.random.choice(max_indices)

        best_action = actions[best_index]
        best_node = children[best_index]
        
        return best_action, best_node
    
    def expand(self, acts: list[int], probs: list[float]) -> None:
        nodes = [Node(self, p, self.node_max, idx) for idx,p in enumerate(probs)]
        
        num_children = len(nodes)
        self.N_values = np.zeros(num_children, dtype=np.int64)
        self.Q_values = np.zeros(num_children, dtype=np.float64)
        self.P_values = np.array(probs, dtype=np.float64)
        
        self._childs = dict(zip(acts,nodes))
        
    def update(self, value: float) -> None:
        self._N += 1
        self._W += value
        
        if(self._parent is not None):
            self._parent.update_child(self.N, self.Q, self.p_idx)
        
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
        cpuct:float = CPUCT, simulation_time: int = 100, p_lr: float = LEARNING_RATE, v_lr:float = LEARNING_RATE
        ):
        self.pv_network = NTuplePVNetwork(board_size, net_list, p_lr, v_lr)
        
        self.env = env
        
        self.board_size = board_size
        
        self.root = Node(None, 1.0, board_size**2, -1)
        
        self.parameter_path = model_path + "/" + PARAMETER_PATH
        self.pv_net_path = model_path + "/" + PV_NET_PATH
        
        self.cpuct = cpuct
        self.simulation_time = simulation_time
        
        self.memo = MCTSMemo()
        
        self.search_player = 0
        
    def PrintMemo(self) -> None:
        self.memo.PrintMemo()
        
    def ResetMemo(self) -> None:
        self.memo.reset()
        
    def Search(self) -> tuple[tuple[int,int], np.ndarray]:
        self.memo.SearchStart()
        
        dat = self.env.backup()
        start_time = time.time()
        
        self.root = Node(None, 1.0, self.board_size**2, -1)
        
        for i in range(self.simulation_time):
            self.__search_once()
            self.env.restore(dat)
            
        
        N_values = self.root.GetChildrenNs()
        actions = self.root.GetLegalActions()
        max_n = np.max(N_values)
        indices = np.where(N_values == max_n)[0]
        
        action = actions[np.random.choice(indices)]
        
        probs = N_values/self.root.N
        indices = np.arange(actions.size)
        
        policy_target = np.full(self.board_size**2, -1, np.float64)
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
        
        p_scores, value = self.pv_network.evaluatePV(board)
        
        legal_move = self.env.GetLegalAction()
        probs = self.__softmax(p_scores[legal_move])
        
        if(node == self.root):
            noise = np.random.dirichlet([ALPHA]*len(probs))
            probs = (1-EPSILON)*probs + EPSILON*noise
        
        node.expand(legal_move,probs)
        
        return value       
        
    def __softmax(self, scores: np.ndarray) -> np.ndarray:
        #スコアを確率に変換
        scores = np.exp(scores - np.max(scores))
        probs = scores / np.sum(scores)
        
        return probs
    
    def train(self, state: np.ndarray, policies: np.ndarray, value: float) -> tuple[float, float]:
        pred_ps, pred_v = self.pv_network.evaluatePV(state)
        
        p_costs,p_loss = self.get_policy_costs_loss(state, policies, pred_ps)
        v_cost,v_loss = self.get_value_cost_loss(state, value, pred_v)
        
        self.pv_network.learnPV(state, p_costs, v_cost, pred_v)
        
        return p_loss, v_loss
        
    def get_policy_costs_loss(self, state: np.ndarray, target: np.ndarray, pred_ps:np.ndarray) -> tuple[np.ndarray, float]:
        legal_moves = np.where(state.flatten() == 0)[0]
        pred_probs = self.__softmax(pred_ps[legal_moves])
        
        target_probs = target[legal_moves]
        
        indices = np.arange(len(legal_moves))
        
        p_costs = np.full(self.board_size**2, 0, dtype=np.float64)
        p_costs[legal_moves[indices]] = pred_probs[indices] - target_probs[indices]
        
        # 交差エントロピー誤差を計算して返す
        # log(0) を防ぐために微小な値(epsilon)を加える
        epsilon = 1e-9
        cross_entropy_loss = -np.average(target_probs * np.log(pred_probs + epsilon))
        
        return p_costs, cross_entropy_loss
    
    def get_value_cost_loss(self, state: np.ndarray, target: float, pred_v:float) -> tuple[float, float]:
        v_cost = pred_v - target
        v_loss = v_cost**2
        
        return v_cost, v_loss
        
    def save(self) -> None:
        self.pv_network.save(self.pv_net_path)
        
    def load(self) -> None:
        self.pv_network.load(self.pv_net_path)
        
        
        
            
        
        
        
            
            
    