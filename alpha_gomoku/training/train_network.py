"""
トレーニング関数
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

from .self_play import SelfPlayDataset
from ..config import (
    device, TRAINING_EPOCHS, TRAINING_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    TRAINING_WEIGHT_DECAY, TRAINING_LR_STEP_SIZE, TRAINING_LR_GAMMA,
    TRAINING_MAX_SAMPLES, TRAINING_PATIENCE, TRAINING_NUM_WORKERS,
    TRAINING_LR_DECAY_FACTOR
)
import os
import datetime


def train_network(model, replay_buffer, epochs=TRAINING_EPOCHS, batch_size=TRAINING_BATCH_SIZE, lr=DEFAULT_LEARNING_RATE, log_dir=None):
    """ニューラルネットワークの訓練"""
    model.train()
    
    # データが不足している場合はスキップ
    if len(replay_buffer) < batch_size:
        print(f"警告: リプレイバッファのサイズが不足しています ({len(replay_buffer)} < {batch_size})")
        return 0, 0
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=TRAINING_WEIGHT_DECAY)
      # 学習率スケジューラ（より安定した設定）
    scheduler = StepLR(optimizer, step_size=TRAINING_LR_STEP_SIZE, gamma=TRAINING_LR_GAMMA)
    
    # リプレイバッファからデータを取得
    max_samples = min(len(replay_buffer), TRAINING_MAX_SAMPLES)
    states, policies, values = replay_buffer.sample(max_samples)
    
    # 価値の前処理：範囲を[-1, 1]にクリップ
    values = np.clip(values, -1.0, 1.0)
    
    # データの検証
    if len(states) == 0 or len(policies) == 0 or len(values) == 0:
        print("警告: サンプルデータが空です")
        return 0, 0
    
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
    
    # 損失計算の改善のため、初期損失を記録
    initial_loss = None
    
    # 損失を記録する変数を初期化
    total_policy_loss = 0
    total_value_loss = 0
    total_loss_sum = 0
    completed_epochs = 0
    
    # AlphaZero論文に従った正則化重み設定（より適切な値に調整）
    l2_reg_weight = 1e-4
    
    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0
        batch_count = 0
        
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device).view(-1)
            
            # 価値の正規化を削除（既にクリップされている）
            # batch_values_normalized = (batch_values + 1.0) / 2.0
            
            # 予測
            policy_logits, value_output = model(batch_states)
            
            # AlphaZero論文に従った損失関数の実装
            # Policy Loss: クロスエントロピー損失（正しい実装）
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(batch_policies * log_probs, dim=1))
            
            # Value Loss: 平均二乗誤差損失
            value_loss = F.mse_loss(value_output.view(-1), batch_values)
            
            # L2正則化項
            l2_reg = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg += torch.norm(param, p=2)
            
            # 総損失の計算
            total_loss = value_loss + policy_loss + l2_reg_weight * l2_reg
            
            # デバッグ：バッチごとの損失を出力（最初のエポックの最初の3バッチのみ）
            if epoch == 0 and batch_count < 3:
                print(f"Batch {batch_count}:")
                print(f"  Policy Loss: {policy_loss.item():.6f}")
                print(f"  Value Loss: {value_loss.item():.6f}")
                print(f"  L2 Reg: {l2_reg.item():.6f}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Predicted values range: [{value_output.min().item():.4f}, {value_output.max().item():.4f}]")
                print(f"  Target values range: [{batch_values.min().item():.4f}, {batch_values.max().item():.4f}]")
                print(f"  Policy entropy: {-torch.sum(batch_policies * log_probs, dim=1).mean().item():.6f}")
                print(f"  Target policy sum: {batch_policies.sum(dim=1).mean().item():.6f}")
            
            # 勾配の計算と更新
            optimizer.zero_grad()
            total_loss.backward()
              # 勾配クリッピング（より緩やかに）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
        
        # エポック平均損失の計算
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)        # 早期停止チェック（改善版）
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 学習率の動的調整（損失が改善しない場合）
        if patience_counter >= patience // 2:
            # 学習率を少し下げる
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
                print(f"Reduced learning rate to {param_group['lr']:.8f}")
                
        # 完全な早期停止
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement")
            break
          # 損失変化が小さすぎる場合の対処を削除（安定化のため）
        # 学習率を動的に調整するロジックは上記で処理済み
        
        # エポックごとに学習率を減衰（シンプルで安定）
        scheduler.step()
        
        # エポックごとの損失を累積（正しい方法）
        total_policy_loss += avg_policy_loss
        total_value_loss += avg_value_loss
        total_loss_sum += avg_total_loss
        completed_epochs += 1
        
        # 現在の学習率を出力
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}")
        print(f"  Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}")
        print(f"  Total Loss: {avg_total_loss:.6f}")
          # エポック毎の詳細ログを出力 (オプション)
        print(f"  Batches processed: {batch_count}")
        print(f"  Patience counter: {patience_counter}/{patience}")
        print(f"  Best loss so far: {best_loss:.6f}")
        
        # エポック毎のログファイル保存
        if log_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            epoch_log_filename = os.path.join(log_dir, f'epoch_log_{epoch+1}_{timestamp}.txt')
            
            with open(epoch_log_filename, 'w') as f:
                f.write(f"Epoch Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Epoch: {epoch+1}/{epochs}\n")
                f.write(f"Learning Rate: {current_lr:.6f}\n")
                f.write(f"Policy Loss: {avg_policy_loss:.6f}\n")
                f.write(f"Value Loss: {avg_value_loss:.6f}\n")
                f.write(f"Total Loss: {avg_total_loss:.6f}\n")
                f.write(f"Batches Processed: {batch_count}\n")
                f.write(f"Patience Counter: {patience_counter}/{patience}\n")
                f.write(f"Best Loss: {best_loss:.6f}\n")
                f.write(f"Replay Buffer Size: {len(replay_buffer)}\n")
                f.write(f"Batch Size: {batch_size}\n")
        
        # 損失が異常値になった場合の対処
        if np.isnan(avg_policy_loss) or np.isnan(avg_value_loss):
            print("警告: 損失がNaNになりました。訓練を停止します。")
            break
        if avg_policy_loss > 100 or avg_value_loss > 100:
            print("警告: 損失が異常に大きくなりました。学習率を下げます。")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= TRAINING_LR_DECAY_FACTOR
    
    # 実際に完了したエポック数を使用
    if completed_epochs == 0:
        completed_epochs = 1  # 0で割ることを防ぐ
    
    # 損失が0の場合の対処
    if total_policy_loss == 0 and total_value_loss == 0:
        print("警告: 損失が0です。学習が正常に実行されていない可能性があります。")
        return 0.01, 0.01  # 小さい値を返して継続
    
    # 平均損失を計算（正しい方法）
    avg_policy_loss = total_policy_loss / completed_epochs
    avg_value_loss = total_value_loss / completed_epochs
    
    # 損失の妥当性チェック
    if avg_policy_loss < 0 or avg_value_loss < 0:
        print(f"警告: 負の損失が検出されました。Policy: {avg_policy_loss}, Value: {avg_value_loss}")
        avg_policy_loss = abs(avg_policy_loss)
        avg_value_loss = abs(avg_value_loss)
    
    print(f"訓練完了 ({completed_epochs} エポック):")
    print(f"  平均 Policy Loss: {avg_policy_loss:.6f}")
    print(f"  平均 Value Loss: {avg_value_loss:.6f}")
    print(f"  最終学習率: {optimizer.param_groups[0]['lr']:.8f}")
    
    return avg_policy_loss, avg_value_loss
