import numpy as np
from GomokuEnv import GomokuEnv
N_IN_ROW = 5  # 五目並べなら5。必要に応じて変更してください。

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
            winner = info.get('win_player', 0) #先手：1, 後手：2
            
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
            #勝ち確定の場合、勝利扱いにする
            if  self.is_winning(self.env):
                winner = self.env.current_player
                # print(f"Game Over: Player {winner} wins")
                # print(f"早期終了盤面")
                self.env.board.PrintBoard()
                winners_z = np.zeros(len(current_players))
                winners_z[np.array(current_players) == winner] = -1.0
                winners_z[np.array(current_players) != winner] = 1.0
                player.reset_player()
                return winners_z, zip(states, mcts_probs, winners_z)
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
            self.env.board.PrintBoard()
            if done:
                winner = info.get('win_player', 0)
                # winner = 3- winner  
                # print(f"Game Over: Player {winner} wins with reward {reward}")
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    # print(f"Winner: Player {winner}")
                    # print(f"Current Players: {current_players}")
                    # print(f"winners_z: {len(winners_z)}")
                    # print(f"len(current_players): {len(current_players)}")
                    # print(f"len(states): {len(states)}")
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                    # print(f"winners_z: {winners_z}")
                player.reset_player()
                # print("返り値チェック")
                # print("winner:", winner)
                # print("states:", states[len(states)-1])
                # # print("mcts_probs:", mcts_probs)
                # print("winners_z:", winners_z)
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
    
    def is_winning(self, env:GomokuEnv):
        """Check if the current player has won or is guaranteed to win in 1 or 2 moves."""
        board_array = env.board.GetBoardInt()
        # print(f"盤面チェック: {board_array}")
        current_player = env.current_player
        opponent = 2 if current_player == 1 else 1
        empty_cells = [(y, x) for y in range(self.board_size) for x in range(self.board_size) if board_array[y][x] == 0]
        # 1手で勝てるかどうかをチェック
        for y, x in empty_cells:
            board_array[y][x] = current_player
            if self.CheckWin(x, y, current_player, N_IN_ROW, board_array):
                board_array[y][x] = 0
                return True
            board_array[y][x] = 0

        # まず、相手が次の一手で勝てるかどうかをチェック
        for oy, ox in empty_cells:
            board_array[oy][ox] = opponent
            if self.CheckWin(ox, oy, opponent, N_IN_ROW, board_array):
                board_array[oy][ox] = 0
                # 相手が次で勝てるなら自分の2手勝ちは成立しない
                return False
            board_array[oy][ox] = 0

        # 2手で、相手がどこに妨害しても勝てるか（ダブルリーチ）効率化
        for idx1 in range(len(empty_cells)):
            y1, x1 = empty_cells[idx1]
            board_array[y1][x1] = current_player
            next_empty = [(y, x) for (y, x) in empty_cells if (y, x) != (y1, x1)]
            win_next = []
            for y2, x2 in next_empty:
                board_array[y2][x2] = current_player
                if self.CheckWin(x2, y2, current_player, N_IN_ROW, board_array):
                    win_next.append((y2, x2))
                board_array[y2][x2] = 0
            if len(win_next) >= 2:
                guaranteed = True
                for block_y, block_x in win_next:
                    board_array[block_y][block_x] = opponent
                    found = False
                    for y2, x2 in win_next:
                        if (y2, x2) == (block_y, block_x):
                            continue
                        if board_array[y2][x2] == 0:
                            board_array[y2][x2] = current_player
                            if self.CheckWin(x2, y2, current_player, N_IN_ROW, board_array):
                                found = True
                            board_array[y2][x2] = 0
                    board_array[block_y][block_x] = 0
                    if not found:
                        guaranteed = False
                        break
                if guaranteed:
                    board_array[y1][x1] = 0
                    return True
            board_array[y1][x1] = 0
        return False
    def CheckWin(self, x: int, y: int, player: int, n_in_row: int, board_array=None) -> bool:
        """
        指定された位置からn_in_row個の石が並んでいるかをチェックします。
        勝利条件を満たしていればTrue、そうでなければFalseを返します。
        board_array: 盤面（2次元リスト）を直接参照する
        """
        if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            return False
        if board_array is None:
            board_array = self.env.board.GetBoardInt()
        stone_type = board_array[y][x]
        if stone_type != player:
            return False
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 右下対角線
            (1, -1)   # 右上対角線
        ]
        for dx, dy in directions:
            count = 1
            # 正方向
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board_array[ny][nx] == player:
                count += 1
                if count >= n_in_row:
                    # print(f"勝利条件を満たす位置: ({x}, {y}) in direction ({dx}, {dy})")
                    return True
                nx += dx
                ny += dy
            # 逆方向
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board_array[ny][nx] == player:
                count += 1
                if count >= n_in_row:
                    # print(f"勝利条件を満たす位置: ({x}, {y}) in direction ({dx}, {dy})")
                    return True
                nx -= dx
                ny -= dy
        return False

    
            
        