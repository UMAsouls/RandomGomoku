import torch
import numpy as np
from rapfi_gomoku import MixNet

class RapfiAgent:
    """
    Rapfiの実装（MixNetモデル）を使用したエージェント。
    get_action()メソッドを実装して他のエージェント（MCTSAgent, RuleBasedAgentなど）と互換性を持つ。
    """
    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: 訓練済みMixNetモデルのパスがあれば指定。Noneの場合は新しいモデルを作成。
            device: 'cuda'または'cpu'。Noneの場合は自動判別。
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Rapfiエージェントを初期化中... デバイス: {self.device}")
        
        # MixNetモデルの初期化
        # 実際の五目並べ盤面サイズに合わせる（15x15を仮定）
        board_size = 15
        self.model = MixNet(
            H=board_size, 
            W=board_size, 
            device=self.device,
            num_possible_patterns_per_type=1024  # デモ用の小さな値、実際は4**11に近い値が必要
        ).to(self.device)
        
        # モデルの重みをロード（もし指定されていれば）
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"モデルをロードしました: {model_path}")
            except Exception as e:
                print(f"モデルのロードに失敗しました: {e}")
        
        self.model.eval()  # 推論モードに設定
    
    def get_action(self, state, current_player):
        """
        現在の盤面状態と現在のプレイヤーに基づいて行動を決定する。
        
        Args:
            state: 盤面状態（2次元リストまたはNumPy配列）
            current_player: 現在のプレイヤー（1または2）
        
        Returns:
            タプル (x, y): 選択された行動の座標
        """
        # stateがリストの場合、NumPy配列に変換
        if isinstance(state, list):
            state = np.array(state)
        
        with torch.no_grad():
            # MixNetモデルを使用して方策と価値を取得
            policy_dist, _ = self.model(state, current_player)
            
            # 方策を1次元配列として取得（形状はboard_size*board_size）
            policy = policy_dist[0].cpu().numpy()
            
            # 盤面サイズを取得
            board_size = state.shape[0]
            
            # 既に石が置かれている場所に対する確率を0にする
            for i in range(board_size):
                for j in range(board_size):
                    if state[i][j] != 0:
                        flat_idx = i * board_size + j
                        policy[flat_idx] = 0.0
            
            # 確率が0でない場所がない場合、ランダムに空いている場所を選択
            if np.sum(policy) == 0:
                empty_positions = []
                for i in range(board_size):
                    for j in range(board_size):
                        if state[i][j] == 0:
                            empty_positions.append((j, i))  # (x, y)形式で格納
                
                if empty_positions:
                    return empty_positions[np.random.randint(0, len(empty_positions))]
                return (0, 0)  # 空いている場所がない場合のフォールバック
            
            # 確率に基づいて行動を選択
            flat_action = np.random.choice(len(policy), p=policy/np.sum(policy))
            
            # 選択された行動を盤面座標に変換（x, y形式）
            y = flat_action // board_size
            x = flat_action % board_size
            
            return (x, y)
