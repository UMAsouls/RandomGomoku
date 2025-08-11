from dataclasses import dataclass

import numpy as np

@dataclass
class EnvBackup:
    board: np.ndarray
    current_player: int
    stone: int
    blackStones: int
    whiteStones: int