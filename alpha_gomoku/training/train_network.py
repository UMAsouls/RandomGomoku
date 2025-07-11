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
from ..config import device


def train_network(model, replay_buffer, epochs=10, batch_size=256, lr=0.001):
    """ニューラルネットワークの訓練"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学習率スケジューラ
    scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
    
    # リプレイバッファからデータを取得
    if len(replay_buffer) < batch_size:
        return 0, 0  # データが十分でない場合はスキップ
    
    # サンプル数を制限して訓練を高速化
    max_samples = min(len(replay_buffer), 3000)
    states, policies, values = replay_buffer.sample(max_samples)
    
    # 価値の前処理：範囲を[-1, 1]にクリップ
    values = np.clip(values, -1.0, 1.0)
    
    # デバッグ：価値の統計を出力
    print(f"価値の統計 - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}, Min: {np.min(values):.4f}, Max: {np.max(values):.4f}")
    
    # データセットとデータローダーの作成
    dataset = SelfPlayDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 早期停止のための変数
    best_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    # 損失を記録する変数を初期化
    total_policy_loss = 0
    total_value_loss = 0
    
    # AlphaZero論文に従った正則化重み設定
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
            
            # 価値を-1から1の範囲から0から1の範囲に変換
            batch_values_normalized = (batch_values + 1.0) / 2.0
            
            # 予測
            policy_logits, value_output = model(batch_states)
            
            # AlphaZero論文に従った損失関数の実装
            # Policy Loss: クロスエントロピー損失
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(batch_policies * log_probs) / batch_states.size(0)
            
            # Value Loss: 平均二乗誤差損失（勝率回帰）
            value_loss = F.mse_loss(value_output.view(-1), batch_values_normalized)
            
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
                print(f"  Predicted values: {value_output[:5].view(-1).detach().cpu().numpy()}")
                print(f"  Target values (normalized): {batch_values_normalized[:5].detach().cpu().numpy()}")
                print(f"  Original target values: {batch_values[:5].detach().cpu().numpy()}")
                print(f"  Policy log_probs sample: {log_probs[0][:10].detach().cpu().numpy()}")
                print(f"  Target policy sample: {batch_policies[0][:10].detach().cpu().numpy()}")
            
            # 勾配の計算と更新
            optimizer.zero_grad()
            total_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
        
        # エポック平均損失の計算
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)
        
        # 早期停止チェック
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # エポックごとに学習率を減衰
        scheduler.step()
        
        # エポックごとの損失を追加
        total_policy_loss += avg_policy_loss
        total_value_loss += avg_value_loss
        
        # 現在の学習率を出力
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}")
        print(f"  Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}")
        print(f"  Total Loss: {avg_total_loss:.6f}")
        
        # 損失が異常値になった場合の対処
        if np.isnan(avg_policy_loss) or np.isnan(avg_value_loss):
            print("警告: 損失がNaNになりました。訓練を停止します。")
            break
        
        if avg_policy_loss > 100 or avg_value_loss > 100:
            print("警告: 損失が異常に大きくなりました。学習率を下げます。")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    
    # 完了したエポック数で平均を計算
    completed_epochs = epoch + 1
    avg_policy_loss = total_policy_loss / completed_epochs
    avg_value_loss = total_value_loss / completed_epochs
    
    return avg_policy_loss, avg_value_loss
