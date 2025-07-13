"""
トレーニング関数
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# matplotlibバックエンドを非インタラクティブに設定（Tkinterエラーを回避）
import matplotlib
matplotlib.use('Agg')  # GUIを使用しないバックエンド
import matplotlib.pyplot as plt

from .self_play import SelfPlayDataset
from ..config import (
    device, TRAINING_EPOCHS, TRAINING_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    TRAINING_WEIGHT_DECAY, TRAINING_LR_STEP_SIZE, TRAINING_LR_GAMMA,
    TRAINING_MAX_SAMPLES, TRAINING_PATIENCE, TRAINING_NUM_WORKERS,
    TRAINING_LR_DECAY_FACTOR
)
import os
import datetime


def set_learning_rate(optimizer, lr):
    """学習率を設定"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_network(model, replay_buffer, epochs=TRAINING_EPOCHS, batch_size=TRAINING_BATCH_SIZE, lr=DEFAULT_LEARNING_RATE, log_dir=None):
    """ニューラルネットワークの訓練"""
    model.train()
    
    # データが不足している場合はスキップ
    if len(replay_buffer) < batch_size:
        print(f"警告: リプレイバッファのサイズが不足しています ({len(replay_buffer)} < {batch_size})")
        return 0, 0
    
    # AlphaZero論文に従った正則化重み設定
    l2_reg_weight = 1e-4

    # オプティマイザの設定
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_weight)
    
    # リプレイバッファからデータを取得
    max_samples = min(len(replay_buffer), TRAINING_MAX_SAMPLES)
    states, policies, values = replay_buffer.sample(max_samples)
    
    # 価値の前処理：範囲を[-1, 1]にクリップ
    values = np.clip(values, -1.0, 1.0)
    
    # データの検証
    if len(states) == 0 or len(policies) == 0 or len(values) == 0:
        print("警告: サンプルデータが空です")
        return 0, 0
    
    # epochごとの損失を記録するリスト
    epoch_policy_losses = []
    epoch_value_losses = []
    epoch_total_losses = []
    epoch_learning_rates = []
    epoch_entropies = []
    
    # デバッグ：価値の統計を出力
    print(f"価値の統計 - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}, Min: {np.min(values):.4f}, Max: {np.max(values):.4f}")
    print(f"ポリシーの統計 - Shape: {policies.shape}, Sum: {np.sum(policies):.4f}")
    
    # データセットとデータローダーの作成
    dataset = SelfPlayDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=TRAINING_NUM_WORKERS)
    
    # 早期停止のための変数
    best_loss = float('inf')
    patience = TRAINING_PATIENCE
    patience_counter = 0
    
    # 損失を記録する変数を初期化
    total_policy_loss = 0
    total_value_loss = 0
    total_loss_sum = 0
    total_entropy = 0
    completed_epochs = 0
    
    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0
        epoch_entropy = 0
        batch_count = 0
        
        # 学習率を設定
        set_learning_rate(optimizer, lr)
        
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device).view(-1)
            
            # 勾配をゼロにリセット
            optimizer.zero_grad()
            
            # 予測（log_softmaxを適用）
            policy_logits, value_output = model(batch_states)
            log_act_probs = F.log_softmax(policy_logits, dim=1)
            
            # 損失計算：AlphaZero論文に従った実装
            # Value Loss: (z - v)^2
            value_loss = F.mse_loss(value_output.view(-1), batch_values)
            
            # Policy Loss: -pi^T * log(p)
            policy_loss = -torch.mean(torch.sum(batch_policies * log_act_probs, 1))
            
            # Total Loss: value_loss + policy_loss（L2正則化はoptimizerに含まれる）
            total_loss = value_loss + policy_loss
            
            # 逆伝播と最適化
            total_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # エントロピー計算（監視用）
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
            )
            
            # デバッグ：バッチごとの損失を出力（最初のエポックの最初の3バッチのみ）
            if epoch == 0 and batch_count < 3:
                print(f"Batch {batch_count}:")
                print(f"  Policy Loss: {policy_loss.item():.6f}")
                print(f"  Value Loss: {value_loss.item():.6f}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Entropy: {entropy.item():.6f}")
                print(f"  Predicted values range: [{value_output.min().item():.4f}, {value_output.max().item():.4f}]")
                print(f"  Target values range: [{batch_values.min().item():.4f}, {batch_values.max().item():.4f}]")
                print(f"  Target policy sum: {batch_policies.sum(dim=1).mean().item():.6f}")
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_entropy += entropy.item()
            batch_count += 1
        
        # エポック平均損失の計算
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_entropy = epoch_entropy / len(dataloader)
        
        # 早期停止チェック
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 学習率の動的調整（損失が改善しない場合）
        if patience_counter >= patience // 2:
            # 学習率を少し下げる
            lr *= 0.9
            print(f"Reduced learning rate to {lr:.8f}")
                
        # 完全な早期停止
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement")
            break
        
        # エポックごとの損失を累積
        total_policy_loss += avg_policy_loss
        total_value_loss += avg_value_loss
        total_loss_sum += avg_total_loss
        total_entropy += avg_entropy
        completed_epochs += 1
        
        # 現在の学習率を出力
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}")
        print(f"  Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}")
        print(f"  Total Loss: {avg_total_loss:.6f}, Entropy: {avg_entropy:.6f}")
        print(f"  Batches processed: {batch_count}")
        print(f"  Patience counter: {patience_counter}/{patience}")
        print(f"  Best loss so far: {best_loss:.6f}")
        
        # epochごとの損失を記録
        epoch_policy_losses.append(avg_policy_loss)
        epoch_value_losses.append(avg_value_loss)
        epoch_total_losses.append(avg_total_loss)
        epoch_learning_rates.append(current_lr)
        epoch_entropies.append(avg_entropy)
        
        # 損失が異常値になった場合の対処
        if np.isnan(avg_policy_loss) or np.isnan(avg_value_loss):
            print("警告: 損失がNaNになりました。訓練を停止します。")
            break
        if avg_policy_loss > 100 or avg_value_loss > 100:
            print("警告: 損失が異常に大きくなりました。学習率を下げます。")
            lr *= TRAINING_LR_DECAY_FACTOR
    
    # 実際に完了したエポック数を使用
    if completed_epochs == 0:
        completed_epochs = 1  # 0で割ることを防ぐ
    
    # 損失が0の場合の対処
    if total_policy_loss == 0 and total_value_loss == 0:
        print("警告: 損失が0です。学習が正常に実行されていない可能性があります。")
        return 0.01, 0.01  # 小さい値を返して継続
    
    # 平均損失を計算
    avg_policy_loss = total_policy_loss / completed_epochs
    avg_value_loss = total_value_loss / completed_epochs
    avg_entropy = total_entropy / completed_epochs
    
    # 損失の妥当性チェック
    if avg_policy_loss < 0 or avg_value_loss < 0:
        print(f"警告: 負の損失が検出されました。Policy: {avg_policy_loss}, Value: {avg_value_loss}")
    avg_policy_loss = abs(avg_policy_loss)
    avg_value_loss = abs(avg_value_loss)
    
    print(f"訓練完了 ({completed_epochs} エポック):")
    print(f"  平均 Policy Loss: {avg_policy_loss:.6f}")
    print(f"  平均 Value Loss: {avg_value_loss:.6f}")
    print(f"  平均 Entropy: {avg_entropy:.6f}")
    print(f"  最終学習率: {optimizer.param_groups[0]['lr']:.8f}")
    
    # epochデータも返すように変更
    epoch_data = {
        'policy_losses': epoch_policy_losses,
        'value_losses': epoch_value_losses,
        'total_losses': epoch_total_losses,
        'learning_rates': epoch_learning_rates,
        'entropies': epoch_entropies
    }
    
    return avg_policy_loss, avg_value_loss, epoch_data


def _save_epoch_graph(current_epoch, total_epochs, policy_losses, value_losses, 
                     total_losses, learning_rates, log_dir, entropies=None):
    """エポックごとの損失をグラフで保存"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # エントロピーがあるかどうかで描画するグラフ数を決定
    if entropies is not None:
        # 5つのサブプロットを持つグラフを作成
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Epoch {current_epoch}/{total_epochs}', fontsize=16)
        
        epochs = list(range(1, len(policy_losses) + 1))
        
        # Policy Loss
        axes[0, 0].plot(epochs, policy_losses, 'b-', marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Policy Loss', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Policy Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Value Loss
        axes[0, 1].plot(epochs, value_losses, 'r-', marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Value Loss', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Value Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Loss
        axes[0, 2].plot(epochs, total_losses, 'g-', marker='o', linewidth=2, markersize=6)
        axes[0, 2].set_title('Total Loss', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Total Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, learning_rates, 'purple', marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Learning Rate', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        axes[1, 1].plot(epochs, entropies, 'orange', marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Policy Entropy', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 最後のサブプロットを非表示
        axes[1, 2].set_visible(False)
        
        # 現在の統計情報を表示
        if len(policy_losses) > 0:
            stats_text = f"Current Stats:\n"
            stats_text += f"Policy Loss: {policy_losses[-1]:.6f}\n"
            stats_text += f"Value Loss: {value_losses[-1]:.6f}\n"
            stats_text += f"Total Loss: {total_losses[-1]:.6f}\n"
            stats_text += f"Learning Rate: {learning_rates[-1]:.6f}\n"
            stats_text += f"Entropy: {entropies[-1]:.6f}"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    else:
        # 4つのサブプロットを持つグラフを作成（従来版）
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Epoch {current_epoch}/{total_epochs}', fontsize=16)
        
        epochs = list(range(1, len(policy_losses) + 1))
        
        # Policy Loss
        axes[0, 0].plot(epochs, policy_losses, 'b-', marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Policy Loss', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Policy Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Value Loss
        axes[0, 1].plot(epochs, value_losses, 'r-', marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Value Loss', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Value Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Loss
        axes[1, 0].plot(epochs, total_losses, 'g-', marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Total Loss', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Total Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, learning_rates, 'purple', marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Learning Rate', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 現在の統計情報を表示
        if len(policy_losses) > 0:
            stats_text = f"Current Stats:\n"
            stats_text += f"Policy Loss: {policy_losses[-1]:.6f}\n"
            stats_text += f"Value Loss: {value_losses[-1]:.6f}\n"
            stats_text += f"Total Loss: {total_losses[-1]:.6f}\n"
            stats_text += f"Learning Rate: {learning_rates[-1]:.6f}"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # ファイル名を作成して保存
    plot_filename = os.path.join(log_dir, f'epoch_progress_{current_epoch}_{timestamp}.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Epoch graph saved: {plot_filename}")


def save_final_training_graph(policy_losses, value_losses, total_losses, learning_rates, log_dir, entropies=None):
    """イテレーション完了時にすべてのepochを含む最終グラフを保存"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # エントロピーがあるかどうかで描画するグラフ数を決定
    if entropies is not None:
        # 5つのサブプロットを持つグラフを作成
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Final Training Results - All Epochs', fontsize=16)
        
        epochs = list(range(1, len(policy_losses) + 1))
        
        # Policy Loss
        axes[0, 0].plot(epochs, policy_losses, 'b-', marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Policy Loss (All Epochs)', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Policy Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Value Loss
        axes[0, 1].plot(epochs, value_losses, 'r-', marker='o', linewidth=2, markersize=4)
        axes[0, 1].set_title('Value Loss (All Epochs)', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Value Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Loss
        axes[0, 2].plot(epochs, total_losses, 'g-', marker='o', linewidth=2, markersize=4)
        axes[0, 2].set_title('Total Loss (All Epochs)', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Total Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, learning_rates, 'purple', marker='o', linewidth=2, markersize=4)
        axes[1, 0].set_title('Learning Rate (All Epochs)', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        axes[1, 1].plot(epochs, entropies, 'orange', marker='o', linewidth=2, markersize=4)
        axes[1, 1].set_title('Policy Entropy (All Epochs)', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 最後のサブプロットを非表示
        axes[1, 2].set_visible(False)
        
        # 最終統計情報を表示
        if len(policy_losses) > 0:
            stats_text = f"Final Stats:\n"
            stats_text += f"Total Epochs: {len(policy_losses)}\n"
            stats_text += f"Final Policy Loss: {policy_losses[-1]:.6f}\n"
            stats_text += f"Final Value Loss: {value_losses[-1]:.6f}\n"
            stats_text += f"Final Total Loss: {total_losses[-1]:.6f}\n"
            stats_text += f"Final Learning Rate: {learning_rates[-1]:.6f}\n"
            stats_text += f"Final Entropy: {entropies[-1]:.6f}\n"
            stats_text += f"Min Policy Loss: {min(policy_losses):.6f}\n"
            stats_text += f"Min Value Loss: {min(value_losses):.6f}\n"
            stats_text += f"Max Entropy: {max(entropies):.6f}"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    else:
        # 4つのサブプロットを持つグラフを作成（従来版）
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Final Training Results - All Epochs', fontsize=16)
        
        epochs = list(range(1, len(policy_losses) + 1))
        
        # Policy Loss
        axes[0, 0].plot(epochs, policy_losses, 'b-', marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Policy Loss (All Epochs)', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Policy Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Value Loss
        axes[0, 1].plot(epochs, value_losses, 'r-', marker='o', linewidth=2, markersize=4)
        axes[0, 1].set_title('Value Loss (All Epochs)', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Value Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Loss
        axes[1, 0].plot(epochs, total_losses, 'g-', marker='o', linewidth=2, markersize=4)
        axes[1, 0].set_title('Total Loss (All Epochs)', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Total Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, learning_rates, 'purple', marker='o', linewidth=2, markersize=4)
        axes[1, 1].set_title('Learning Rate (All Epochs)', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 最終統計情報を表示
        if len(policy_losses) > 0:
            stats_text = f"Final Stats:\n"
            stats_text += f"Total Epochs: {len(policy_losses)}\n"
            stats_text += f"Final Policy Loss: {policy_losses[-1]:.6f}\n"
            stats_text += f"Final Value Loss: {value_losses[-1]:.6f}\n"
            stats_text += f"Final Total Loss: {total_losses[-1]:.6f}\n"
            stats_text += f"Final Learning Rate: {learning_rates[-1]:.6f}\n"
            stats_text += f"Min Policy Loss: {min(policy_losses):.6f}\n"
            stats_text += f"Min Value Loss: {min(value_losses):.6f}"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    # ファイル名を作成して保存
    plot_filename = os.path.join(log_dir, f'final_training_results_{timestamp}.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Final training graph saved: {plot_filename}")
    return plot_filename

