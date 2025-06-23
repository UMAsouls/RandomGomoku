from N_Tuple import NTupleQAgent
from GomokuEnv import GomokuEnv

import numpy as np

game = GomokuEnv(train_target="both")
agent = NTupleQAgent(board_size=game.board_size)

episodes = 2

for episode in range(episodes):
    game.reset()  # ゲームのリセット
    state = game.get_board()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換
        #print(set_act)
        next_state, reward, done, _ = game.step(set_act)
        
        next_state = np.array(next_state)  # 次の状態もNumPy配列

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        #game.render()

    print(f"Episode: {episode}, Total Reward: {total_reward}")

