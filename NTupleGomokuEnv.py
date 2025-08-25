import numpy as np

from const import EnvBackup

from GomokuEnv import GomokuEnv
from RandomGomoku import GetBoard, IBoard,Stone

from Interfaces import IEnv

class NTupleGomokuEnv(GomokuEnv):
    def __init__(self, board_size=19, train_target="first"):
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
            
        return self.GetBoard_CurrentPlayer(), reward, done, a2
        
    def GetBoard(self):
        return self.board.GetBoardInt()
        
    def GetBoard_Player1(self):
        return self.board.GetBoardInt()
    
    def GetBoard_Player2(self):
        return self.board.GetBoardOppose()
    
    def GetBoard_CurrentPlayer(self) -> np.ndarray:
        if self.current_player == 2 and self.train_player == 0:
            return self.board.GetBoardOppose()
        else:
            return self.board.GetBoardInt()
        
    def GetCurrentPlayer(self):
        return self.current_player
        
    def GetLegalAction(self) -> np.ndarray:
        b = self.board.GetBoardInt()
        
        y,x = np.where(b == 0)
        arr = y*self.board_size + x
        
        return arr
    
    # 盤面をキャリッジリターンで出力し、カーソルを改行分だけ上げる  
    def Animation(self):
        print("\r{0}{1}".format(self.board.StringBoard(), f"\033[{self.board_size+1}A"), end = "", flush=True)
    
    # アニメーション終了処理
    # カーソルをアニメーションエリアの分だけ下げる   
    def AnimationEnd(self):
        print(f"\033[{self.board_size+1}B")
        
    def backup(self):
        s = 1 if self.stone == Stone.BLACK else 2
        b = self.board.GetBoardInt()
        o = self.board.GetBoardOppose()
        return EnvBackup(b, o, self.current_player, s, self.blackStones, self.whiteStones)
        
    def restore(self, data: EnvBackup):
        self.board.SetBoard(data.board.copy(), data.oppose.copy())
        self.current_player = data.current_player
        self.stone = Stone.BLACK if data.stone == 1 else Stone.WHITE
        self.blackStones = data.blackStones
        self.whiteStones = data.whiteStones
        
    def SetBoard(self, board: np.ndarray):
        board1 = np.zeros((self.board_size, self.board_size), dtype=np.int64)
        board2 = np.zeros((self.board_size, self.board_size), dtype=np.int64)
        
        ix1,iy1 = np.where(board == 1)
        board1[ix1,iy1] = 1
        
        ix2,iy2 = np.where(board == 2)
        board2[ix2,iy2] = 1
        
        self.board.SetBoard(board1, board2)
        
        self.blackStones = len(ix1)
        self.whiteStones = len(ix2)
        
    def PrintBoard(self):
        self.board.PrintBoard()
        

if __name__ == "__main__":
    board = NTupleGomokuEnv()
    
    #board.board.SetStone(0,0, Stone.BLACK)
    board.board.PrintBoard()
    
    print(board.GetLegalAction())