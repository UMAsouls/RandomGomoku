"""
トレーニングユーティリティ関数
"""

import numpy as np
import torch
# matplotlibバックエンドを非インタラクティブに設定（Tkinterエラーを回避）
import matplotlib
matplotlib.use('Agg')  # GUIを使用しないバックエンド
import matplotlib.pyplot as plt
from collections import deque


class LossStabilizer:
    """損失の安定化を支援するクラス"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.policy_losses = deque(maxlen=window_size)
        self.value_losses = deque(maxlen=window_size)
        self.moving_avg_policy = []
        self.moving_avg_value = []
        
    def add_loss(self, policy_loss, value_loss):
        """損失を追加し、移動平均を計算"""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        
        # 移動平均を計算
        if len(self.policy_losses) >= 3:  # 最低3つのデータポイントが必要
            self.moving_avg_policy.append(np.mean(self.policy_losses))
            self.moving_avg_value.append(np.mean(self.value_losses))
        
    def get_smoothed_losses(self):
        """平滑化された損失を取得"""
        return self.moving_avg_policy, self.moving_avg_value
    
    def is_loss_stable(self, threshold=0.01):
        """損失が安定しているかチェック"""
        if len(self.policy_losses) < self.window_size:
            return False
        
        # 最近の損失の標準偏差をチェック
        policy_std = np.std(self.policy_losses)
        value_std = np.std(self.value_losses)
        
        return policy_std < threshold and value_std < threshold
    
    def detect_loss_explosion(self, threshold=10.0):
        """損失の爆発を検出"""
        if len(self.policy_losses) < 2:
            return False
        
        current_policy = self.policy_losses[-1]
        current_value = self.value_losses[-1]
        
        # 前回の損失と比較
        prev_policy = self.policy_losses[-2]
        prev_value = self.value_losses[-2]
        
        policy_ratio = current_policy / (prev_policy + 1e-8)
        value_ratio = current_value / (prev_value + 1e-8)
        
        return policy_ratio > threshold or value_ratio > threshold


class AdaptiveLearningRate:
    """適応的学習率調整クラス"""
    
    def __init__(self, initial_lr=0.001, patience=5, factor=0.5, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def step(self, loss):
        """損失に基づいて学習率を調整"""
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            # 学習率を下げる
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self.patience_counter = 0
                print(f"学習率を {self.current_lr:.6f} に調整しました")
                return True
        
        return False
    
    def get_lr(self):
        """現在の学習率を取得"""
        return self.current_lr


def smooth_loss_curve(losses, window_size=5):
    """損失曲線を平滑化"""
    if len(losses) < window_size:
        return losses
    
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window_size // 2)
        end = min(len(losses), i + window_size // 2 + 1)
        smoothed.append(np.mean(losses[start:end]))
    
    return smoothed


def plot_stable_loss_history(policy_losses, value_losses, iterations, save_path=None):
    """安定化された損失履歴をプロット"""
    plt.figure(figsize=(15, 10))
    
    # 元の損失をプロット
    plt.subplot(2, 2, 1)
    plt.plot(iterations, policy_losses, 'b-', alpha=0.3, label='Policy Loss (Raw)')
    plt.plot(iterations, value_losses, 'r-', alpha=0.3, label='Value Loss (Raw)')
    
    # 平滑化された損失をプロット
    if len(policy_losses) > 5:
        smooth_policy = smooth_loss_curve(policy_losses, window_size=5)
        smooth_value = smooth_loss_curve(value_losses, window_size=5)
        plt.plot(iterations, smooth_policy, 'b-', linewidth=2, label='Policy Loss (Smoothed)')
        plt.plot(iterations, smooth_value, 'r-', linewidth=2, label='Value Loss (Smoothed)')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 対数スケールでもプロット
    plt.subplot(2, 2, 2)
    plt.semilogy(iterations, policy_losses, 'b-', alpha=0.5, label='Policy Loss')
    plt.semilogy(iterations, value_losses, 'r-', alpha=0.5, label='Value Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 損失の変化率をプロット
    if len(policy_losses) > 1:
        plt.subplot(2, 2, 3)
        policy_changes = np.diff(policy_losses)
        value_changes = np.diff(value_losses)
        plt.plot(iterations[1:], policy_changes, 'b-', label='Policy Loss Change')
        plt.plot(iterations[1:], value_changes, 'r-', label='Value Loss Change')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Iterations')
        plt.ylabel('Loss Change')
        plt.title('Loss Change Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 損失の移動平均をプロット
    if len(policy_losses) > 10:
        plt.subplot(2, 2, 4)
        window = min(10, len(policy_losses) // 3)
        ma_policy = smooth_loss_curve(policy_losses, window_size=window)
        ma_value = smooth_loss_curve(value_losses, window_size=window)
        plt.plot(iterations, ma_policy, 'b-', linewidth=2, label=f'Policy Loss (MA-{window})')
        plt.plot(iterations, ma_value, 'r-', linewidth=2, label=f'Value Loss (MA-{window})')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Moving Average Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"安定化された損失グラフを保存しました: {save_path}")
    
    return plt
