from N_Tuple import NTupleQAgent
from GomokuEnv import GomokuEnv

import numpy as np

import time

BOARD_SIZE = 15  # ボードのサイズ

game = GomokuEnv(BOARD_SIZE,train_target="both")
agent = NTupleQAgent(board_size=game.board_size)

episodes = 1000

for episode in range(episodes):
    game.reset()  # ゲームのリセット
    state = game.get_board()
    done = False
    total_reward = 0
    
    select_time = 0
    step_time = 0
    update_time = 0
    
    t0 = time.time()
    while not done:
        t1 = time.time()
        action = agent.select_action(state)
        set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換
        select_time += time.time() - t1
        
        t2 = time.time()
        next_state, reward, done, _ = game.step(set_act)
        next_state = np.array(next_state)  # 次の状態もNumPy配列
        step_time += time.time() - t2

        t3 = time.time()
        agent.update(state, action, reward, next_state, done)
        update_time += time.time() - t3
        
        state = next_state
        total_reward += reward
        #game.render()
        
    episode_time = time.time() - t0
    print(f"Select Time: {select_time:.4f}s, Step Time: {step_time:.4f}s, Update Time: {update_time:.4f}s, Episode Time: {episode_time:.4f}s")
    
    print(f"Episode: {episode}, Total Reward: {total_reward}")

