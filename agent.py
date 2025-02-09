
import random

class RandomAgent:
    def __init__(self):
        pass
    
    #state=盤面の状態
    #empty_positions=空いてるマスのリスト作成
    def get_action(self, state):
        empty_positions = [(j, i) for i in range(19) for j in range(19) if state[i][j] == 0]
        return random.choice(empty_positions)
    
class RuleBasedAgent:
    def __init__(self):
        pass
    
    def get_action(self, state, player_stone):
        already_placed = [(j, i) for i in range(19) for j in range(19) if state[i][j] != 0]
        positions = []
        for i, j in already_placed:
            for i_offset in range(-1, 2):
                for j_offset in range(-1, 2):
                    if i_offset == 0 and j_offset == 0:
                        continue
                    if 0 <= i + i_offset < 19 and 0 <= j + j_offset < 19 and state[i + i_offset][j + j_offset] == 0:
                        positions.append((j + j_offset,i + i_offset ))
    
        random.shuffle(positions)  # 探索順序をランダム化
        
        for i, j in positions:
            if self.check_win(state, i, j, player_stone):
                return (j, i)
        
        for i, j in positions:
            if self.check_win(state, i, j, 3 - player_stone):
                return (j, i)
        
        for i, j in positions:
            if self.check_three_in_a_row(state, i, j, 3 - player_stone):
                return (j, i)
        
        for i, j in positions:
            if self.check_three_in_a_row(state, i, j, player_stone):
                return (j, i)
        
        return random.choice(positions)
        
    def check_three_in_a_row(self, state, i, j, player_stone):
        """相手が3つ並べているかどうかを判定し、その手を阻止する"""
        if state[i][j] != 0:  # 既に石があるかを確認
            return False
        # 一時的に石を置いて3つ並びができるか確認
        state[i][j] = player_stone
        three_in_a_row = (
            self.check_consecutive(state, player_stone, 3)
        )
        state[i][j] = 0  # 石を戻す
        return three_in_a_row
        
        
    def check_consecutive(self, state, player_stone, count):
        """縦・横・斜めに指定された個数が連続しているかを確認"""
        # 縦方向
        for i in range(19 - count + 1):
            for j in range(19):
                if all(state[i + k][j] == player_stone for k in range(count)):
                    return True
        # 横方向
        for i in range(19):
            for j in range(19 - count + 1):
                if all(state[i][j + k] == player_stone for k in range(count)):
                    return True
        # 斜め（左上→右下）
        for i in range(19 - count + 1):
            for j in range(19 - count + 1):
                if all(state[i + k][j + k] == player_stone for k in range(count)):
                    return True
        # 斜め（右上→左下）
        for i in range(19 - count + 1):
            for j in range(count - 1, 19):
                if all(state[i + k][j - k] == player_stone for k in range(count)):
                    return True
        return False
    
    def check_win(self, state, i, j, player_stone):
        if state[i][j] != 0:  # 既に石があるかを確認
            return False
        #player_stone=1:先手,2:後手
        #石を打って勝てるかどうかを判定する
        #石を打つ
        state[i][j] = player_stone
        #勝敗を判定
        win = self.check_win_vertical(state, player_stone) or self.check_win_horizontal(state, player_stone) or self.check_win_diagonal(state, player_stone)
        #石を戻す
        state[i][j] = 0
        return win
    
    def check_win_vertical(self, state, player_stone):
        for i in range(15):
            for j in range(19):
                if all(state[i + k][j] == player_stone for k in range(5)):
                    return True
        return False
    
    def check_win_horizontal(self, state, player_stone):
        for i in range(19):
            for j in range(15):
                if all(state[i][j + k] == player_stone for k in range(5)):
                    return True
        return False
    
    def check_win_diagonal(self, state, player_stone):
        for i in range(15):
            for j in range(15):
                if all(state[i + k][j + k] == player_stone for k in range(5)):
                    return True
        for i in range(15):
            for j in range(4, 19):
                if all(state[i + k][j - k] == player_stone for k in range(5)):
                    return True
        return False
    
import random

class MinimaxAgent:
    def __init__(self, depth=2):
        self.depth = depth
    
    def get_action(self, state, player_stone):
        best_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')

        # 状態に基づいてすでに置かれている石の座標を記録
        already_placed = self.get_already_placed(state)
        
        positions = self.generate_possible_moves(state, already_placed)

        random.shuffle(positions)

        for i, j in positions:
            if state[i][j] == 0:  # 空きマスのみ探索
                state[i][j] = player_stone  # 仮に石を置く
                score = self.minimax(state, self.depth - 1, False, alpha, beta, player_stone)
                state[i][j] = 0  # 元に戻す
                if score > best_score:
                    best_score = score
                    best_action = (j, i)
        
        return best_action if best_action else random.choice([(i, j) for i in range(19) for j in range(19) if state[i][j] == 0])

    def minimax(self, state, depth, is_maximizing, alpha, beta, player_stone):
        opponent_stone = 3 - player_stone
        if self.check_win(state, player_stone):
            return 10000
        if self.check_win(state, opponent_stone):
            return -10000
        if depth == 0 or all(state[i][j] != 0 for i in range(19) for j in range(19)):
            return self.evaluate_state(state, player_stone)

        already_placed = self.get_already_placed(state)
        positions = self.generate_possible_moves(state, already_placed)

        random.shuffle(positions)

        if is_maximizing:
            best_score = float('-inf')
            for i, j in positions:
                state[i][j] = player_stone
                #端はスコア低めにする
                if i==0 or i==18 or j==0 or j==18:
                    score -= 30
                score = self.minimax(state, depth - 1, False, alpha, beta, player_stone)
                state[i][j] = 0
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for i, j in positions:
                state[i][j] = opponent_stone
                score = self.minimax(state, depth - 1, True, alpha, beta, player_stone)
                state[i][j] = 0
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

    def evaluate_state(self, state, player_stone):
        opponent_stone = 3 - player_stone
        score = 0
        
        if self.check_win(state, player_stone):
            return 10000
        if self.check_win(state, opponent_stone):
            return -10000
        
        # 自分の連続石に応じたスコア加算
        consecutive_count = self.count_consecutive(state, player_stone)
        score += consecutive_count[2] * 5
        score += consecutive_count[3] * 20
        score += consecutive_count[4] * 100
        score += consecutive_count[5] * 10000  # 勝利状態

        # 相手の連続石に応じたスコア減算
        score -= consecutive_count[3] * 500
        score -= consecutive_count[4] * 1000
        score -= consecutive_count[5] * 10000  # 相手の勝利状態は大幅ペナルティ

        return score

    def count_consecutive(self, state, player_stone):
        """ 連続した石の数を方向別にカウントする """
        counts = {2: 0, 3: 0, 4: 0, 5: 0}
        
        # 横、縦、斜めの連続石を確認
        for i in range(19):
            for j in range(19):
                if state[i][j] == player_stone:
                    for count in [2, 3, 4, 5]:
                        # 横、縦、斜めをチェック
                        for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                            if self.check_consecutive(state, i, j, di, dj, count, player_stone):
                                counts[count] += 1
        return counts

    def check_consecutive(self, state, i, j, di, dj, count, player_stone):
        """ 連続石が指定数分あるか確認 """
        for k in range(count):
            ni, nj = i + di * k, j + dj * k
            if not (0 <= ni < 19 and 0 <= nj < 19 and state[ni][nj] == player_stone):
                return False
        return True

    def get_already_placed(self, state):
        """ すでに置かれた石の位置を取得 """
        return [(i, j) for i in range(19) for j in range(19) if state[i][j] != 0]

    def generate_possible_moves(self, state, already_placed):
        """ 隣接する空きマスを生成 """
        positions = []
        for i, j in already_placed:
            for i_offset in range(-1, 2):
                for j_offset in range(-1, 2):
                    if i_offset == 0 and j_offset == 0:
                        continue
                    if 0 <= i + i_offset < 19 and 0 <= j + j_offset < 19 and state[i + i_offset][j + j_offset] == 0:
                        positions.append((i + i_offset, j + j_offset))
        return positions

    def check_win(self, state, player_stone):
        """ 5連があるかを確認 """
        return self.count_consecutive(state, player_stone)[5] > 0
