"""
AlphaZeroトレーナー
"""

import os
import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import concurrent.futures

from ..network.dual_network import DualNetwork
from ..mcts.mcts import MCTS
from ..mcts.game_utils import (
    detect_winning_move, detect_blocking_move, augment_data, 
    get_legal_moves, is_terminal, get_winner_value
)
from .replay_buffer import ReplayBuffer
from .train_network import train_network
from ..config import device, SIMULATIONS


class AlphaZero:
    """AlphaZeroの実装"""
    
    def __init__(self, board_size=19, num_iterations=100, num_self_play_games=64,
                 checkpoint_dir='models', log_dir='logs', initial_lr=0.0005):
        self.board_size = board_size
        self.num_iterations = num_iterations
        self.num_self_play_games = num_self_play_games
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.initial_lr = initial_lr
        
        # ディレクトリの作成
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # モデルの初期化
        self.model = DualNetwork(board_size).to(device)
        
        # リプレイバッファ
        self.replay_buffer = ReplayBuffer(capacity=40000)
        
        # タイムスタンプの作成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # モデルのベース名を設定
        self.model_base_name = f'alpha_gomoku_{board_size}'
        
        # モデルのチェックポイントパス
        self.model_path = os.path.join(checkpoint_dir, f'{self.model_base_name}_{timestamp}.pth')
        
        # 時間ベースのログディレクトリを作成
        self.timestamp_log_dir = os.path.join(log_dir, f'log_{timestamp}')
        os.makedirs(self.timestamp_log_dir, exist_ok=True)
        
        # 損失履歴を記録するリスト
        self.policy_loss_history = []
        self.value_loss_history = []
        self.iterations = []
        self.lr_history = []
        
        # 既存のモデルをロード
        self._load_existing_model()
    
    def _load_existing_model(self):
        """既存のモデルをロード"""
        if os.path.exists(self.model_path):
            try:
                print(f"モデルを読み込みました: {self.model_path}")
            except:
                print("新規モデルを初期化します")
        else:
            # 最新のモデルファイルを検索
            model_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith(self.model_base_name) and f.endswith('.pth')]
            if model_files:
                # 最新のモデルをロード
                latest_model = os.path.join(self.checkpoint_dir, sorted(model_files)[-1])
                try:
                    self.model.load_state_dict(torch.load(latest_model, map_location=device))
                    print(f"最新のモデルを読み込みました: {latest_model}")
                    self.model_path = latest_model
                except:
                    print("新規モデルを初期化します")
            else:
                print("新規モデルを初期化します")
            
            # 初期モデルの保存
            self.save_model("initial")
    
    def save_model(self, stage, iteration=None):
        """モデルを保存"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイル名の生成
        if iteration is not None:
            model_filename = f'{self.model_base_name}_iter{iteration}_{stage}_{timestamp}.pth'
        else:
            model_filename = f'{self.model_base_name}_{stage}_{timestamp}.pth'
            
        save_path = os.path.join(self.checkpoint_dir, model_filename)
        
        # モデルの保存
        torch.save(self.model.state_dict(), save_path)
        print(f"モデルを保存しました: {save_path}")
        
        # 最新のモデルパスを更新
        self.model_path = save_path
        return save_path
    
    def train(self):
        """AlphaZeroの訓練ループ"""
        for iteration in range(self.num_iterations):
            start_time = time.time()
            print(f"\nイテレーション {iteration+1}/{self.num_iterations}")
            
            # 学習率の調整
            current_lr = self.initial_lr * (0.5 ** (iteration // 20))
            print(f"現在の初期学習率: {current_lr:.6f}")
            self.lr_history.append(current_lr)
            
            current_iter = iteration + 1
            
            # 1. 自己対戦でデータ生成
            print("自己対戦でデータを生成中...")
            self._generate_self_play_data()
            
            # 自己対戦後のモデルを保存
            self.save_model("after_selfplay", iteration+1)
            
            # 2. ニューラルネットワークの訓練
            print("ニューラルネットワークを訓練中...")
            policy_loss, value_loss = train_network(self.model, self.replay_buffer, lr=current_lr)
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # 損失履歴に追加
            if policy_loss > 0:
                self.policy_loss_history.append(policy_loss)
                self.value_loss_history.append(value_loss)
                self.iterations.append(current_iter)
            
            # 訓練後のモデルを保存
            self.save_model("trained", iteration+1)
            
            # グラフを保存
            self._plot_loss_history(iteration=current_iter)
            
            # 3. ログファイルに保存
            self._save_training_log(iteration+1, current_lr, policy_loss, value_loss, start_time)
            
            iteration_time = time.time() - start_time
            print(f"イテレーション完了: {iteration_time:.2f} 秒")
        
        # 最終モデルを保存
        self.save_model("final")
    
    def _save_training_log(self, iteration, lr, policy_loss, value_loss, start_time):
        """トレーニング情報をログファイルに保存"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.timestamp_log_dir, f'training_log_{iteration}_{timestamp}.txt')
        
        with open(log_filename, 'w') as f:
            f.write(f"Training Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Iteration: {iteration}/{self.num_iterations}\n")
            f.write(f"Initial Learning Rate: {lr:.6f}\n")
            f.write(f"Policy Loss: {policy_loss:.6f}\n")
            f.write(f"Value Loss: {value_loss:.6f}\n")
            f.write(f"Replay Buffer Size: {len(self.replay_buffer)}\n")
            f.write(f"Self-Play Games: {self.num_self_play_games}\n")
            f.write(f"Board Size: {self.board_size}\n")
            f.write(f"Training Duration: {time.time() - start_time:.2f} seconds\n")
        
        print(f"トレーニング情報をログに保存しました: {log_filename}")
    
    def _plot_loss_history(self, final=False, stage=None, iteration=None):
        """損失の履歴をグラフ化して保存"""
        plt.figure(figsize=(15, 15))
        
        # グラフタイトル
        title_suffix = ""
        if final:
            title_suffix = " (Final)"
        elif iteration:
            title_suffix = f" (Iteration {iteration})"
        
        # 学習率の履歴データ
        lr_iterations = list(range(1, len(self.lr_history) + 1))
        
        total_subplots = 3 if len(self.policy_loss_history) > 0 else 1
        
        # Policy Lossグラフ
        if len(self.policy_loss_history) > 0:
            plt.subplot(total_subplots, 1, 1)
            plt.plot(self.iterations, self.policy_loss_history, 'b-', marker='o')
            plt.title(f'Policy Loss History{title_suffix}')
            plt.xlabel('Iteration')
            plt.ylabel('Policy Loss')
            plt.grid(True)
            
            # Value Lossグラフ
            plt.subplot(total_subplots, 1, 2)
            plt.plot(self.iterations, self.value_loss_history, 'r-', marker='o')
            plt.title(f'Value Loss History{title_suffix}')
            plt.xlabel('Iteration')
            plt.ylabel('Value Loss')
            plt.grid(True)
        
        # 学習率の履歴
        plt.subplot(total_subplots, 1, total_subplots)
        plt.plot(lr_iterations, self.lr_history, 'g-', marker='o')
        plt.title(f'Learning Rate History{title_suffix}')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        # 現在の訓練状況を表示
        plt.figtext(0.5, 0.01, 
                   f"Board Size: {self.board_size}, Buffer Size: {len(self.replay_buffer)}, Games/Iter: {self.num_self_play_games}", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # ファイル名の作成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if iteration:
            plot_filename = os.path.join(self.timestamp_log_dir, f'loss_history_iter{iteration}_{timestamp}.png')
        else:
            plot_filename = os.path.join(self.timestamp_log_dir, f'loss_history_final_{timestamp}.png')
        
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        
        print(f"損失のグラフを保存しました: {plot_filename}")
    
    def _generate_self_play_data(self):
        """自己対戦データの生成"""
        # 現在の最新のモデルを使用
        torch.save(self.model.state_dict(), self.model_path)
        
        # マルチプロセッシングの準備
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        
        # 共有メモリの準備
        manager = mp.Manager()
        shared_buffer = manager.list()
        
        # プロセス数を動的に設定
        available_cpu = mp.cpu_count()
        num_processes = min(int(available_cpu * 0.6), self.num_self_play_games, 8)
        num_processes = max(1, num_processes)
        
        games_per_process = self.num_self_play_games // num_processes
        remainder = self.num_self_play_games % num_processes
        
        print(f"並列プロセス数: {num_processes}, プロセスあたりの基本ゲーム数: {games_per_process}")
        
        # プロセス作成と開始
        start_idx = 0
        processes = []
        for i in range(num_processes):
            process_games = games_per_process + (1 if i < remainder else 0)
            end_idx = start_idx + process_games
            
            p = ctx.Process(
                target=self._self_play_process,
                args=(self.model_path, self.board_size, shared_buffer, 
                      range(start_idx, end_idx), result_queue)
            )
            p.start()
            processes.append(p)
            start_idx = end_idx
        
        # 進捗表示
        pbar = tqdm(total=self.num_self_play_games, desc="自己対戦")
        completed_games = 0
        
        while completed_games < self.num_self_play_games:
            if not result_queue.empty():
                game_idx, result = result_queue.get()
                completed_games += 1
                pbar.update(1)
        
        pbar.close()
        
        # 全プロセス終了待ち
        for p in processes:
            p.join()
        
        # リプレイバッファにデータを追加
        batch_size = 1000
        buffer_list = list(shared_buffer)
        for i in range(0, len(buffer_list), batch_size):
            batch = buffer_list[i:i+batch_size]
            for state, policy, value in batch:
                self.replay_buffer.add(state, policy, value)
        
        print(f"リプレイバッファサイズ: {len(self.replay_buffer)}")
    
    def _self_play_process(self, model_path, board_size, shared_buffer, game_indices, result_queue):
        """自己対戦プロセス"""
        from GomokuEnv import GomokuEnv
        
        # モデルのロード
        model = DualNetwork(board_size)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"モデルロードエラー: {e}")
        
        model.to(device)
        model.eval()
        
        # MCTSの設定
        base_simulations = max(50, SIMULATIONS)
        total_spaces = board_size * board_size
        mcts = MCTS(model, num_simulations=base_simulations, use_gumbel=True, gumbel_scale=0.02)
        
        # 一時的なバッファ
        temp_buffer = []
        max_temp_size = 2000
        
        for game_idx in game_indices:
            try:
                # 環境の初期化
                env = GomokuEnv(board_size=board_size)
                
                # 残り空きマスの数に応じてシミュレーション回数を調整
                def adjust_simulations(state):
                    remaining_spaces = np.sum(np.array(state) == 0)
                    ratio = remaining_spaces / total_spaces
                    adjusted_sims = max(30, min(int(base_simulations * (0.5 + 0.5 * ratio)), base_simulations))
                    return adjusted_sims
                
                mcts.num_simulations = base_simulations
                
                game_memory = []
                state = env.board.GetBoardInt()
                
                done = False
                current_player = 1
                move_count = 0
                last_move = None
                
                # ゲーム実行
                while not done and move_count < board_size * board_size:
                    move_count += 1
                    state_array = np.array(state)
                    
                    # シミュレーション回数を調整
                    adjusted_sims = adjust_simulations(state_array)
                    mcts.num_simulations = adjusted_sims
                    
                    # 現在のプレイヤーを特定
                    player = 1 if np.sum(state_array == 1) == np.sum(state_array == -1) else -1
                    
                    # 勝利パターンチェック
                    winning_move = detect_winning_move(state_array.copy(), player, board_size)
                    if winning_move is not None:
                        one_hot = np.zeros(board_size * board_size)
                        one_hot[winning_move] = 1.0
                        mcts_policy = one_hot
                        mcts_value = 1.0
                    else:
                        # 負け防止パターンチェック
                        blocking_move = detect_blocking_move(state_array.copy(), player, board_size)
                        if blocking_move is not None:
                            one_hot = np.zeros(board_size * board_size)
                            one_hot[blocking_move] = 1.0
                            mcts_policy = one_hot
                            mcts_value = 0.0
                        else:
                            # 通常のMCTSで方策と価値を取得
                            try:
                                mcts_policy, mcts_value = mcts.search(state_array, last_move)
                            except Exception as e:
                                print(f"MCTS検索中にエラーが発生: {e}")
                                
                                # フォールバック: ランダム方策
                                legal_moves = [(i % board_size, i // board_size) 
                                              for i in range(board_size * board_size) 
                                              if state_array[i // board_size][i % board_size] == 0]
                                if legal_moves:
                                    mcts_policy = np.zeros(board_size * board_size)
                                    for move in legal_moves:
                                        idx = move[1] * board_size + move[0]
                                        mcts_policy[idx] = 1.0 / len(legal_moves)
                                    mcts_value = 0.0
                                else:
                                    break
                    
                    # 訓練データの保存
                    game_memory.append((state_array.copy(), mcts_policy.copy(), mcts_value, player))
                    
                    # 行動の選択
                    try:
                        action_idx = np.random.choice(len(mcts_policy), p=mcts_policy)
                        action = (action_idx % board_size, action_idx // board_size)
                    except Exception as e:
                        print(f"行動選択中にエラーが発生: {e}")
                        legal_moves = [(i % board_size, i // board_size) 
                                      for i in range(board_size * board_size) 
                                      if state_array[i // board_size][i % board_size] == 0]
                        if legal_moves:
                            action = random.choice(legal_moves)
                        else:
                            break
                    
                    # 環境での行動実行
                    try:
                        next_state, reward, done, _ = env.step(action)
                        last_move = action
                        state = next_state.cpu().numpy()
                        current_player *= -1
                    except Exception as e:
                        print(f"環境ステップ実行中にエラーが発生: {e}")
                        break
                
                # ゲーム終了後の処理
                final_game_result = reward.item() if 'reward' in locals() else 0
                
                # 価値の範囲チェック
                value_samples = []
                
                # 各手のデータに対して処理
                for i, (hist_state, hist_policy, hist_mcts_value, hist_player) in enumerate(game_memory):
                    # ゲーム結果に基づく最終価値を計算
                    if final_game_result == 0:  # 引き分け
                        final_value = 0.0
                    else:
                        # プレイヤー視点でのゲーム結果を計算
                        final_value = final_game_result if hist_player == 1 else -final_game_result
                    
                    # MCTSの価値と最終結果を重み付け平均で組み合わせ
                    game_progress = i / len(game_memory)
                    result_weight = 0.2 + 0.6 * game_progress
                    mcts_weight = 1.0 - result_weight
                    
                    training_value = mcts_weight * hist_mcts_value + result_weight * final_value
                    
                    # 価値を[-1, 1]の範囲にクリップ
                    training_value = np.clip(training_value, -1.0, 1.0)
                    
                    # デバッグ用のサンプリング
                    if len(value_samples) < 5:
                        value_samples.append(training_value)
                    
                    # データ拡張
                    aug_prob = max(0.1, 0.3 - (len(shared_buffer) / 50000))
                    
                    if np.random.random() < aug_prob:
                        try:
                            augmented_states, augmented_policies = augment_data(hist_state, hist_policy, board_size)
                            for i, (aug_state, aug_policy) in enumerate(zip(augmented_states[:3], augmented_policies[:3])):
                                temp_buffer.append((aug_state, aug_policy, training_value))
                        except Exception as e:
                            print(f"データ拡張中にエラーが発生: {e}")
                            temp_buffer.append((hist_state, hist_policy, training_value))
                    else:
                        temp_buffer.append((hist_state, hist_policy, training_value))
                    
                    # 一時バッファがサイズを超えたら転送
                    if len(temp_buffer) >= max_temp_size:
                        try:
                            for item in temp_buffer:
                                shared_buffer.append(item)
                            temp_buffer = []
                        except Exception as e:
                            print(f"バッファ転送中にエラーが発生: {e}")
                            temp_buffer = []
                
                # デバッグ出力
                if len(value_samples) > 0:
                    print(f"ゲーム{game_idx}: 最終結果={final_game_result}, "
                          f"価値サンプル={[f'{v:.3f}' for v in value_samples]}")
                
                # 結果をキューに送信
                result_queue.put((game_idx, final_game_result))
                
            except Exception as e:
                print(f"ゲーム{game_idx}実行中にエラーが発生: {e}")
                result_queue.put((game_idx, 0))
        
        # 残りのデータを転送
        try:
            for item in temp_buffer:
                shared_buffer.append(item)
        except Exception as e:
            print(f"最終バッファ転送中にエラーが発生: {e}")
    
    def play_against_human(self):
        """人間との対戦"""
        from GomokuEnv import GomokuEnv
        
        env = GomokuEnv(board_size=self.board_size)
        state = env.board.GetBoardInt()
        
        # MCTSの初期化
        base_simulations = 2000
        total_spaces = self.board_size * self.board_size
        mcts = MCTS(self.model, num_simulations=base_simulations, use_gumbel=False)
        
        # 動的シミュレーション調整
        def adjust_simulations(state):
            remaining_spaces = np.sum(np.array(state) == 0)
            ratio = remaining_spaces / total_spaces
            adjusted_sims = max(int(base_simulations * ratio), int(base_simulations * 0.2))
            return adjusted_sims
        
        done = False
        human_first = input("先手で始めますか？ (y/n): ").lower() == 'y'
        
        if not human_first:
            # AIの手番
            state_array = np.array(state)
            
            # シミュレーション回数を調整
            mcts.num_simulations = adjust_simulations(state_array)
            
            current_player = 1 if np.sum(state_array == 1) == np.sum(state_array == -1) else -1
            
            # 勝利パターンチェック
            winning_move = detect_winning_move(state_array.copy(), current_player, self.board_size)
            if winning_move is not None:
                print("AIが勝利パターンを検出しました")
                action_idx = winning_move
            else:
                # 負け防止パターンチェック
                blocking_move = detect_blocking_move(state_array.copy(), current_player, self.board_size)
                if blocking_move is not None:
                    print("AIが負け防止パターンを検出しました")
                    action_idx = blocking_move
                else:
                    # 通常のMCTSで手を決定
                    mcts_policy, _ = mcts.search(state_array)
                    action_idx = np.argmax(mcts_policy)
            
            action = (action_idx % self.board_size, action_idx // self.board_size)
            state, reward, done, _ = env.step(action)
            state = state.cpu().numpy()
            env.render()
        
        while not done:
            # 人間の手番
            try:
                x = int(input(f"列 (0-{self.board_size-1}): "))
                y = int(input(f"行 (0-{self.board_size-1}): "))
                if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size or state[y][x] != 0:
                    print("無効な手です。再入力してください。")
                    continue
                action = (x, y)
                state, reward, done, _ = env.step(action)
                state = state.cpu().numpy()
                env.render()
                
                if done:
                    print("あなたの勝ちです！")
                    break
                
                # AIの手番
                state_array = np.array(state)
                
                # シミュレーション回数を調整
                mcts.num_simulations = adjust_simulations(state_array)
                
                current_player = 1 if np.sum(state_array == 1) == np.sum(state_array == -1) else -1
                
                # 勝利パターンチェック
                winning_move = detect_winning_move(state_array.copy(), current_player, self.board_size)
                if winning_move is not None:
                    print("AIが勝利パターンを検出しました")
                    action_idx = winning_move
                    
                    # 勝利確定手を実行
                    action = (action_idx % self.board_size, action_idx // self.board_size)
                    state, reward, done, _ = env.step(action)
                    state = state.cpu().numpy()
                    env.render()
                    
                    if done:
                        print("AIの勝利です！")
                    break
                else:
                    # 負け防止パターンチェック
                    blocking_move = detect_blocking_move(state_array.copy(), current_player, self.board_size)
                    if blocking_move is not None:
                        print("AIが負け防止パターンを検出しました")
                        action_idx = blocking_move
                    else:
                        # 通常のMCTSで手を決定
                        mcts_policy, _ = mcts.search(state_array)
                        action_idx = np.argmax(mcts_policy)
                
                action = (action_idx % self.board_size, action_idx // self.board_size)
                state, reward, done, _ = env.step(action)
                state = state.cpu().numpy()
                env.render()
                
                if done:
                    print("AIの勝利です！")
                
            except ValueError:
                print("数値を入力してください")
                continue
