import numpy as np
from GomokuEnv import GomokuEnv

class Game(object):
    def __init__(self, env:GomokuEnv, board_size=19):
        self.env = env
        self.board_size = env.board_size
        self.board = env.board
        self.current_player = env.current_player
        self.lastmove = env.lastmove

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        2人のプレイヤー間でゲームを開始します。
        player1, player2: プレイヤーオブジェクト
        start_player: 先手プレイヤー（0=player1, 1=player2）
        is_shown: ゲームの進行状況を表示するかどうか
        """
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                           'or 1 (player2 first)')
        
        # 新しいゲーム環境を初期化
        self.env = GomokuEnv(board_size=self.board_size)
        
        # start_playerに基づいてプレイヤーの割り当てを決定
        if start_player == 0:
            # player1が先手（プレイヤー1）
            p1, p2 = 1, 2
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2: player2}
        else:
            # player2が先手（プレイヤー1）
            p1, p2 = 1, 2
            player1.set_player_ind(p2)
            player2.set_player_ind(p1)
            players = {p1: player2, p2: player1}
        
        if is_shown:
            print(f"Game start: Player {p1} vs Player {p2}")
            print(f"先手: Player {p1 if start_player == 0 else p2}")
        

        
        # ゲームループ
        move_count = 0
        max_moves = self.board_size * self.board_size  # 最大手数制限
        done = False
        while not done and move_count < max_moves:
            # 現在のプレイヤーを決定
            current_player = self.env.current_player
            
            if is_shown:
                print(f"\n--- 手番 {move_count + 1}: Player {current_player} ---")
            
            # プレイヤーから行動を取得
            if current_player in players:
                try:
                    move = players[current_player].get_action(self.env)
                    if is_shown:
                        print(f"Player {current_player} の手: {move}")
                except Exception as e:
                    if is_shown:
                        print(f"Player {current_player} でエラーが発生: {e}")
                    # エラーが発生した場合、相手の勝利とする
                    if current_player == 1:
                        return -1  # Player 2の勝利
                    else:
                        return 1   # Player 1の勝利
            else:
                if is_shown:
                    print(f"不正なプレイヤー: {current_player}")
                break
            
            # 行動を実行
            try:
                _, reward, done, info = self.env.step(move)
                move_count += 1
                if is_shown and hasattr(self.env.board, 'PrintBoard'):
                    self.env.board.PrintBoard()
                
            except Exception as e:
                if is_shown:
                    print(f"無効な手: {move}, エラー: {e}")
                # 無効な手を打った場合、相手の勝利とする
                if current_player == 1:
                    return -1  # Player 2の勝利
                else:
                    return 1   # Player 1の勝利
        
        # ゲーム終了処理
        if done:
            winner = info.get('which_player', 0) #先手：1, 後手：2
            
            self.env.board.PrintBoard()
            if is_shown:
                print(f"\nゲーム終了: Player {winner} の勝利!")
                
            if winner == 1:
                print("Player 1の勝利")
                return 1
            elif winner == 2:
                print("Player 2の勝利")
                return -1

        else:
            # 最大手数に達した場合は引き分け
            if is_shown:
                print(f"\nゲーム終了: 最大手数 {max_moves} に達しました。引き分け!")
            return 0
        
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        セルフプレイを開始します。
        player: MCTSプレイヤー
        is_shown: ゲームの進行状況を表示するかどうか
        temp: 温度パラメータ（探索のランダム性を制御）
        """
        self.env = GomokuEnv(board_size=self.board_size)
        states, mcts_probs, current_players = [], [], []
        while True:
            move, mcts_prob = player.get_action(self.env, temp=temp, return_prob=True)
            
            # ボードが満杯で有効な手がない場合の処理
            if move is None:
                # 引き分け処理
                winners_z = np.zeros(len(current_players))
                player.reset_player()
                return winners_z, zip(states, mcts_probs, winners_z)
            
            #TODO:ボードがあっているか確認
            # print(self.env.board.GetBoardInt())
            #　現在のプレイヤーの石のみのstateを返す、値はどちらのプレイヤーでも1
            
                        
            # print(board_array)
            states.append(self.make_state(self.env.board.GetBoardInt()))
            mcts_probs.append(mcts_prob)
            current_players.append(self.env.current_player)
            # print("current_players")
            # print(self.env.current_player)
            
            _,reward,done,info= self.env.step(move)

            if done:
                winner = info.get('win_player', 0)
                print(f"Game Over: Player {winner} wins with reward {reward}")
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                    # print(f"winners_z: {winners_z}")
                player.reset_player()
                print("返り値チェック")
                print("states:", states)
                # print("mcts_probs:", mcts_probs)
                print("winners_z:", winners_z)
                return winners_z, zip(states, mcts_probs, winners_z)
    def make_state(self, board_array:list[list[int]]):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = np.zeros((4, self.board_size, self.board_size))
        if board_array is not None:
            # 現在のプレイヤーの石の位置を取得
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board_array[i][j] == self.env.current_player:
                        square_state[0][i][j] = 1.0
                    elif board_array[i][j] != 0:  # 相手の石
                        square_state[1][i][j] = 1.0
            
            # 最後の手の位置を表示
            if self.env.lastmove is not None:
                last_x, last_y = self.env.lastmove
                square_state[2][last_y][last_x] = 1.0
            
            # 手数が偶数の場合（後手番）を表示
            board_array_np = np.array(board_array)
            total_stones = np.sum(board_array_np != 0)
            # print(f"Total stones placed: {total_stones}")
            # print(f"Board array: {board_array}")
            if total_stones % 2 == 0:
                square_state[3][:, :] = 1.0  # indicate the colour to play
        
        return square_state
            
        