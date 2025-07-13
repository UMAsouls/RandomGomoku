"""
MCTSノード
"""

class MCTSNode:
    """モンテカルロ木探索のノード"""
    
    def __init__(self, prior=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.win_count = 0  # 勝利数を追跡
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def win_rate(self):
        """勝率（w/n）を計算"""
        if self.visit_count == 0:
            return 0
        return self.win_count / self.visit_count

    def update(self, value):
        """シミュレーション結果でノードを更新"""
        self.visit_count += 1
        self.value_sum += value
        
        # 勝利数の更新（value > 0の場合を勝利とする）
        if value > 0:
            self.win_count += 1
