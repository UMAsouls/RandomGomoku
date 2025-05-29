import torch
import numpy as np
import time
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from alpha_gomoku import DualNetwork, MCTS
import GomokuEnv

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomPlayer:
    """ランダムに手を選ぶプレイヤー"""
    def __init__(self, board_size):
        self.board_size = board_size
    
    def get_action(self, state):
        """有効な手からランダムに選択"""
        valid_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if state[y][x] == 0:  # 空きマス
                    valid_moves.append((x, y))
        
        if not valid_moves:
            return None  # 有効な手がない
        
        return random.choice(valid_moves)

def simulate_games(num_games=100, board_size=7, model_path='models/alpha_gomoku_7_20250521_134646.pth'):
    """AlphaGomokuとランダムプレイヤーの対戦をシミュレーション"""
    # モデルの初期化とロード
    model = DualNetwork(board_size).to(device)
    if os.path.exists(model_path):
        try:
            print(f"モデルをロードしています: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"モデルを読み込みました: {model_path}")
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {str(e)}")
            return
    else:
        print(f"モデルが見つかりません: {model_path}")
        return
    
    model.eval()  # 評価モード
    
    # MCTSの初期化
    mcts = MCTS(model, num_simulations=800)  # シミュレーション回数
    
    # ランダムプレイヤーの初期化
    random_player = RandomPlayer(board_size)
    
    results = {"alpha_first": {"wins": 0, "losses": 0, "draws": 0},
               "random_first": {"wins": 0, "losses": 0, "draws": 0}}
    
    # 進捗バーの表示
    for game in tqdm(range(num_games), desc="対戦進行中"):
        # 先手と後手を交互に
        alpha_first = game % 2 == 0
        
        env = GomokuEnv.GomokuEnv(board_size=board_size)
        state = env.board.GetBoardInt()
        done = False
        
        game_result = None
        
        while not done:
            if (alpha_first and env.current_player == 1) or (not alpha_first and env.current_player == 2):
                # AlphaGomokuの手番
                mcts_policy = mcts.search(np.array(state))
                action_idx = np.argmax(mcts_policy)
                action = (action_idx % board_size, action_idx // board_size)
            else:
                # ランダムプレイヤーの手番
                action = random_player.get_action(state)
            
            # 環境を進める
            next_state, reward, done, _ = env.step(action)
            state = next_state.cpu().numpy()
            
            if done:
                # ゲーム終了時の処理
                winner = 3-env.current_player
                if alpha_first:
                    if winner == 1:  # AlphaGomokuの勝利 (先手)
                        results["alpha_first"]["wins"] += 1
                    else:  # ランダムプレイヤーの勝利 (後手)
                        results["alpha_first"]["losses"] += 1
                else:
                    if winner == 2:  # AlphaGomokuの勝利 (後手)
                        results["random_first"]["wins"] += 1
                    else:  # ランダムプレイヤーの勝利 (先手)
                        results["random_first"]["losses"] += 1
    
    # 結果の集計
    total_games = num_games
    alpha_wins = results["alpha_first"]["wins"] + results["random_first"]["wins"]
    alpha_losses = results["alpha_first"]["losses"] + results["random_first"]["losses"]
    
    # 結果の表示
    print("\n===== 対戦結果 =====")
    print(f"総対戦数: {total_games}")
    print(f"AlphaGomokuの勝利: {alpha_wins} ({alpha_wins/total_games*100:.2f}%)")
    print(f"ランダムプレイヤーの勝利: {alpha_losses} ({alpha_losses/total_games*100:.2f}%)")
    
    print("\n--- AlphaGomoku先手の場合 ---")
    alpha_first_games = num_games // 2 + (num_games % 2)
    print(f"対戦数: {alpha_first_games}")
    print(f"勝利: {results['alpha_first']['wins']} ({results['alpha_first']['wins']/alpha_first_games*100:.2f}%)")
    print(f"敗北: {results['alpha_first']['losses']} ({results['alpha_first']['losses']/alpha_first_games*100:.2f}%)")
    
    print("\n--- ランダムプレイヤー先手の場合 ---")
    random_first_games = num_games // 2
    print(f"対戦数: {random_first_games}")
    print(f"勝利: {results['random_first']['wins']} ({results['random_first']['wins']/random_first_games*100:.2f}%)")
    print(f"敗北: {results['random_first']['losses']} ({results['random_first']['losses']/random_first_games*100:.2f}%)")
    
    # グラフで可視化
    try:
        plot_results(results, alpha_wins, alpha_losses, total_games)
    except Exception as e:
        print(f"グラフ作成エラー: {str(e)}")

def plot_results(results, alpha_wins, alpha_losses, total_games):
    """結果をグラフで表示"""
    labels = ['全体', 'Alpha先手', 'Random先手']
    alpha_first_games = total_games // 2 + (total_games % 2)
    random_first_games = total_games // 2
    
    alpha_win_rates = [
        alpha_wins / total_games * 100,
        results['alpha_first']['wins'] / alpha_first_games * 100,
        results['random_first']['wins'] / random_first_games * 100
    ]
    
    random_win_rates = [
        alpha_losses / total_games * 100,
        results['alpha_first']['losses'] / alpha_first_games * 100,
        results['random_first']['losses'] / random_first_games * 100
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    alpha_bars = ax.bar(x - width/2, alpha_win_rates, width, label='AlphaGomoku勝率')
    random_bars = ax.bar(x + width/2, random_win_rates, width, label='Random勝率')
    
    ax.set_ylabel('勝率 (%)')
    ax.set_title('AlphaGomokuとランダムプレイヤーの対戦結果')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 値をバーの上に表示
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(alpha_bars)
    add_labels(random_bars)
    
    plt.tight_layout()
    plt.savefig('alpha_vs_random_results.png')
    plt.show()

if __name__ == "__main__":
    # パラメータ設定
    NUM_GAMES = 100
    BOARD_SIZE = 7
    MODEL_PATH = 'models/alpha_gomoku_7_20250524_084916.pth'
    
    # シミュレーション実行
    start_time = time.time()
    simulate_games(num_games=NUM_GAMES, board_size=BOARD_SIZE, model_path=MODEL_PATH)
    end_time = time.time()
    
    print(f"\n実行時間: {end_time - start_time:.2f}秒")
