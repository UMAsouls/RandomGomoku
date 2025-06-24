from GomokuEnv import GomokuEnv

class NTupleGomokuEnv(GomokuEnv):
    def __init__(self, board_size=19, train_target="first", n_tuple_size=4):
        super().__init__(board_size, train_target)


    def step(self, action):
        pass