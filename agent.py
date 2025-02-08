
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
        positions = [(i, j) for i in range(19) for j in range(19) if state[i][j] == 0]
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
    def __init__(self, depth=1):
        self.depth = depth  # 最大探索深さ
    
    def get_action(self, state, player_stone):
        best_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')

        for i in range(19):
            for j in range(19):
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
            return 1000 if is_maximizing else -1000
        if self.check_win(state, opponent_stone):
            return -1000 if is_maximizing else 1000
        if depth == 0 or all(state[i][j] != 0 for i in range(19) for j in range(19)):
            return self.evaluate_state(state, player_stone)

        if is_maximizing:
            best_score = float('-inf')
            for i in range(19):
                for j in range(19):
                    if state[i][j] == 0:
                        state[i][j] = player_stone
                        score = self.minimax(state, depth - 1, False, alpha, beta, player_stone)
                        state[i][j] = 0
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:  # βカット
                            break
            return best_score
        else:
            best_score = float('inf')
            for i in range(19):
                for j in range(19):
                    if state[i][j] == 0:
                        state[i][j] = opponent_stone
                        score = self.minimax(state, depth - 1, True, alpha, beta, player_stone)
                        state[i][j] = 0
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        if beta <= alpha:  # αカット
                            break
            return best_score

    def evaluate_state(self, state, player_stone):
        """ 改良された評価関数 """
        opponent_stone = 3 - player_stone
        score = 0

        # 自分の連続石に応じたスコア加算
        score += self.count_consecutive(state, player_stone, 2) * 5
        score += self.count_consecutive(state, player_stone, 3) * 20
        score += self.count_consecutive(state, player_stone, 4) * 100
        score += self.count_consecutive(state, player_stone, 5) * 10000  # 勝利状態

        # 相手の連続石に応じたスコア減算（相手の強い状態を阻止）
        score -= self.count_consecutive(state, opponent_stone, 3) * 25
        score -= self.count_consecutive(state, opponent_stone, 4) * 200
        score -= self.count_consecutive(state, opponent_stone, 5) * 10000  # 相手の勝利状態は大幅ペナルティ

        # 盤面中央に近いほど高く評価（x, y の距離を考慮）
        for i in range(19):
            for j in range(19):
                if state[i][j] == player_stone:
                    score += 10 - max(abs(i - 9), abs(j - 9))  # 中央 (9, 9) に近いほど高得点

        return score


    def count_consecutive(self, state, player_stone, count):
        """縦・横・斜めに指定された個数が連続しているパターンを数える"""
        total = 0
        for i in range(19):
            for j in range(19):
                if self.check_consecutive(state, i, j, player_stone, count):
                    total += 1
        return total

    def check_consecutive(self, state, i, j, player_stone, count):
        """ 縦・横・斜めに指定された個数が連続しているかを確認 """
        if j + count <= 19 and all(state[i][j + k] == player_stone for k in range(count)):
            return True
        if i + count <= 19 and all(state[i + k][j] == player_stone for k in range(count)):
            return True
        if i + count <= 19 and j + count <= 19 and all(state[i + k][j + k] == player_stone for k in range(count)):
            return True
        if i + count <= 19 and j - count >= -1 and all(state[i + k][j - k] == player_stone for k in range(count)):
            return True
        return False

    def check_win(self, state, player_stone):
        """ 5連があるかを確認 """
        return self.count_consecutive(state, player_stone, 5) > 0
