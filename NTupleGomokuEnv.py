from GomokuEnv import GomokuEnv
from RandomGomoku import GetFastBoard, FastBoard

class NTupleGomokuEnv(GomokuEnv):
    def __init__(self, board_size=19, train_target="first", n_tuple_size=4):
        super().__init__(board_size, train_target)
        
        self.board: FastBoard = GetFastBoard(board_size, board_size)


    def step(self, action):
        a1, reward, done, a2 = super().step(action)
        
        return self.board.GetBoardWB(), reward, done, a2