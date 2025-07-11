"""
ゲームルールユーティリティ
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def check_win_pattern_numba(state, x, y, player, board_size):
    """勝利条件チェックのNumba最適化版"""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1  # 自分自身
        
        # 正方向
        for i in range(1, 5):  # 最大4つ先まで確認
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < board_size and 0 <= ny < board_size and state[ny][nx] == player:
                count += 1
            else:
                break
                
        # 負方向
        for i in range(1, 5):  # 最大4つ先まで確認
            nx, ny = x - dx * i, y - dy * i
            if 0 <= nx < board_size and 0 <= ny < board_size and state[ny][nx] == player:
                count += 1
            else:
                break
            
        if count >= 5:
            return True
            
    return False


@jit(nopython=True, cache=True)
def get_legal_moves(state):
    """盤面の合法手を返すNumba最適化版"""
    board_size = state.shape[0]
    legal_moves = []
    for y in range(board_size):
        for x in range(board_size):
            if state[y][x] == 0:
                legal_moves.append(y * board_size + x)
    return legal_moves


def get_winner_value(state, board_size):
    """勝者に基づく価値を返す"""
    # 水平方向
    for y in range(board_size):
        for x in range(board_size - 4):
            if state[y][x] != 0 and all(state[y][x] == state[y][x+i] for i in range(5)):
                return state[y][x]  # 勝者の値 (1 or -1)
                
    # 垂直方向
    for y in range(board_size - 4):
        for x in range(board_size):
            if state[y][x] != 0 and all(state[y+i][x] == state[y][x] for i in range(5)):
                return state[y][x]
                
    # 右下がり対角線
    for y in range(board_size - 4):
        for x in range(board_size - 4):
            if state[y][x] != 0 and all(state[y+i][x+i] == state[y][x] for i in range(5)):
                return state[y][x]
                
    # 左下がり対角線
    for y in range(board_size - 4):
        for x in range(4, board_size):
            if state[y][x] != 0 and all(state[y+i][x-i] == state[y][x] for i in range(5)):
                return state[y][x]
    
    # 引き分け
    return 0


def detect_winning_move(state, player, board_size):
    """勝利パターンを検出（一手で勝てるか）"""
    legal_moves = get_legal_moves(state)
    for move in legal_moves:
        y, x = divmod(move, board_size)
        # 一時的に石を置いてみる
        state[y][x] = player
        
        # 勝利条件チェック
        if check_win_pattern_numba(state, x, y, player, board_size):
            # 元に戻す
            state[y][x] = 0
            return move
        
        # 元に戻す
        state[y][x] = 0
    
    return None


def detect_blocking_move(state, player, board_size):
    """相手の勝利を阻止する手を検出"""
    # 相手のプレイヤー番号
    opponent = -player
    
    # 相手が次の手で勝てる場所を検出
    return detect_winning_move(state, opponent, board_size)


def has_open_four(state, move, board_size):
    """四つ並んでいて相手が干渉していない状況を検出する"""
    x, y = move % board_size, move // board_size
    if state[y][x] != 0:  # すでに石が置かれている場合
        return False
    
    # プレイヤーを特定
    player = 1 if np.sum(state == 1) == np.sum(state == -1) else -1
    
    # 安全に操作するためのコピーを作成
    temp_state = state.copy()
    temp_state[y][x] = player
    
    # 方向ベクトル定義: 横、縦、右下、左下
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1  # 自分自身
        
        # 両端の開放状態を確認
        pos_open = False
        neg_open = False
        
        # 正方向に連続する石を数える
        nx, ny = x + dx, y + dy
        while 0 <= nx < board_size and 0 <= ny < board_size and temp_state[ny][nx] == player:
            count += 1
            nx += dx
            ny += dy
        
        # 正方向の端が空いているか確認
        pos_open = (0 <= nx < board_size and 0 <= ny < board_size and temp_state[ny][nx] == 0)
        
        # 負方向に連続する石を数える
        nx, ny = x - dx, y - dy
        while 0 <= nx < board_size and 0 <= ny < board_size and temp_state[ny][nx] == player:
            count += 1
            nx -= dx
            ny -= dy
        
        # 負方向の端が空いているか確認
        neg_open = (0 <= nx < board_size and 0 <= ny < board_size and temp_state[ny][nx] == 0)
        
        # 四つ並びで少なくとも片方の端が空いている場合
        if count == 4 and (pos_open or neg_open):
            return True
    
    return False


def is_winning_move(state, move, board_size):
    """与えられた手が勝利に繋がるかチェックする軽量版"""
    x, y = move % board_size, move // board_size
    if state[y][x] != 0:  # すでに石が置かれている場合
        return False
    
    # プレイヤーを特定
    player = 1 if np.sum(state == 1) == np.sum(state == -1) else -1
    
    # 方向ベクトル定義: 横、縦、右下、左下
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1  # 自分自身
        
        # 正方向
        nx, ny = x + dx, y + dy
        while 0 <= nx < board_size and 0 <= ny < board_size and state[ny][nx] == player:
            count += 1
            nx += dx
            ny += dy
        
        # 負方向
        nx, ny = x - dx, y - dy
        while 0 <= nx < board_size and 0 <= ny < board_size and state[ny][nx] == player:
            count += 1
            nx -= dx
            ny -= dy
        
        if count >= 5:
            return True
            
    return False


def is_terminal(state, board_size):
    """ゲームが終了状態かどうかをチェック"""
    # 勝者がいる場合は終了状態
    if get_winner_value(state, board_size) != 0:
        return True
        
    # 盤面に空きマスがない場合も終了状態（引き分け）
    for y in range(board_size):
        for x in range(board_size):
            if state[y][x] == 0:
                return False
    
    return True


def augment_data(state, policy, board_size):
    """データ拡張：回転と反転による盤面とポリシーの対称変換を行う"""
    # 状態とポリシーを盤面形式に変形
    policy_grid = policy.reshape(board_size, board_size)
    
    # 変換されたデータを格納するリスト
    augmented_states = []
    augmented_policies = []
    
    # 元の状態とポリシーを追加
    augmented_states.append(state.copy())
    augmented_policies.append(policy.copy())
    
    # 90度回転
    rot90_state = np.rot90(state, k=1)
    rot90_policy = np.rot90(policy_grid, k=1).flatten()
    augmented_states.append(rot90_state)
    augmented_policies.append(rot90_policy)
    
    # 180度回転
    rot180_state = np.rot90(state, k=2)
    rot180_policy = np.rot90(policy_grid, k=2).flatten()
    augmented_states.append(rot180_state)
    augmented_policies.append(rot180_policy)
    
    # 270度回転
    rot270_state = np.rot90(state, k=3)
    rot270_policy = np.rot90(policy_grid, k=3).flatten()
    augmented_states.append(rot270_state)
    augmented_policies.append(rot270_policy)
    
    # 水平反転
    flip_h_state = np.fliplr(state)
    flip_h_policy = np.fliplr(policy_grid).flatten()
    augmented_states.append(flip_h_state)
    augmented_policies.append(flip_h_policy)
    
    # 垂直反転
    flip_v_state = np.flipud(state)
    flip_v_policy = np.flipud(policy_grid).flatten()
    augmented_states.append(flip_v_state)
    augmented_policies.append(flip_v_policy)
    
    # 対角線反転（左上から右下）
    diag_state = np.transpose(state)
    diag_policy = np.transpose(policy_grid).flatten()
    augmented_states.append(diag_state)
    augmented_policies.append(diag_policy)
    
    # 反対角線反転（右上から左下）
    anti_diag_state = np.transpose(np.fliplr(np.flipud(state)))
    anti_diag_policy = np.transpose(np.fliplr(np.flipud(policy_grid))).flatten()
    augmented_states.append(anti_diag_state)
    augmented_policies.append(anti_diag_policy)
    
    return augmented_states, augmented_policies
