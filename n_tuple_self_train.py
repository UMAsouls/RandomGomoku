from N_Tuple import NTupleQAgent
from NTupleGomokuEnv import NTupleGomokuEnv

import numpy as np

import time

BOARD_SIZE = 15  # ボードのサイズ

MODEL_DIR = "NTupleQModel"
MODEL_NAME = "NTupleQModel_mini"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

NETS = [10]

game = NTupleGomokuEnv(BOARD_SIZE,train_target="both")
agent = NTupleQAgent(board_size=game.board_size, model_path=MODEL_PATH, net_list=NETS)

episodes = 100000
save_rate = 5000

for episode in range(episodes):
    game.reset()  # ゲームのリセット
    state = game.board.GetBoardInt()
    bef_state = np.nan
    bef_action = None
    done = False
    total_reward = 0
    
    select_time = 0
    game_step_time = 0
    update_time = 0
    
    step_num = 0
    
    t0 = time.time()
    while not done:
        t1 = time.time()
        action = agent.select_action(state)
        set_act = (action % game.board_size, action // game.board_size) #x, yのタプルに変換
        select_time += time.time() - t1
        
        t2 = time.time()
        next_state, reward, done, _ = game.step(set_act)
        next_oppose_state = game.OpposeBoard()
        game_step_time += time.time() - t2

        t3 = time.time()
        # 相手の行動を環境による変化と考える
        # するとtdtargetは相手の行動結果を対象に作らなければならない
        # doneのときは(bef_actionをした側を自分として)相手が勝ったから、この手を打ったら負けると学習できる
        if(bef_state is not np.nan or bef_action is not None):
            agent.update(bef_state, bef_action, -reward, next_state, done)
        
        # 試合終了した際はnext_stateが意味を無くすので、reward単体をtdtargetとすることができる
        # この手を打ったら勝てると学習できる
        if(done):
            agent.update(state, action, reward, next_state, done)
        update_time += time.time() - t3
        
        
        bef_state = state
        state = next_state
        
        bef_action = action
        
        total_reward += reward
        step_num += 1
        #game.Animation()
        
        
    #game.AnimationEnd()
    episode_time = time.time() - t0
    print(f"Select Time: {select_time:.4f}s, Game Step Time: {game_step_time:.4f}s, Update Time: {update_time:.4f}s, Episode Time: {episode_time:.4f}s")
    print(f"Step: {step_num}step, Step Time: {(episode_time/step_num*1000):.4f}ms")
    
    print(f"Episode: {episode}, Total Reward: {total_reward}")

    if (episode % save_rate == 0):
        agent.save()

