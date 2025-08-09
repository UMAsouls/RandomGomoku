from GomokuEnv import GomokuEnv
from RandomGomoku import GetBoard, IBoard

class NTupleGomokuEnv(GomokuEnv):
    def __init__(self, board_size=19, train_target="first", n_tuple_size=4):
        super().__init__(board_size, train_target)
        
        self.board: IBoard = GetBoard(board_size, board_size)
        

    def step(self, action):
        a1, reward, done, a2 = super().step(action)

        if done == 1:
            if self.current_player == 1 or self.train_player == 0:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        
        if self.current_player == 2 and self.train_player == 0:
            return self.board.GetBoardOppose(), reward, done, a2
        else:
            return self.board.GetBoardInt(), reward, done, a2
    
    def OpposeBoard(self):
        if self.current_player == 2 and self.train_player == 0:
            return self.board.GetBoardInt()
        else:
            return self.board.GetBoardOppose()
    
    # 盤面をキャリッジリターンで出力し、カーソルを改行分だけ上げる  
    def Animation(self):
        print("\r{0}{1}".format(self.board.StringBoard(), f"\033[{self.board_size+1}A"), end = "", flush=True)
    
    # アニメーション終了処理
    # カーソルをアニメーションエリアの分だけ下げる   
    def AnimationEnd(self):
        print(f"\033[{self.board_size+1}B")