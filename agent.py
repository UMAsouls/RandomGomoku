
import random

class RandomAgent:
    def __init__(self):
        pass
    
    #state=盤面の状態
    #empty_positions=空いてるマスのリスト作成
    def get_action(self, state):
        empty_positions = [(j, i) for i in range(19) for j in range(19) if state[i][j] == 0]
        return random.choice(empty_positions)
    