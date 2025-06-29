import torch
import time

SIZE = 10000
TEST = 10000

def slow():
    tensor = torch.zeros(SIZE, SIZE).cuda()

    start_time = time.time()
    # 遅い例：Pythonのループとインデックス指定による個別の代入
    for i in range(TEST):
        for j in range(TEST):
            tensor[i, j] = i + j
    end_time = time.time()
    print(tensor)
    print(f"個別の代入にかかった時間: {end_time - start_time:.4f}秒")
    
def fast1():
    tensor = torch.zeros(SIZE, SIZE)

    # 高速な例1：スライシングとブロードキャスト
    start_time = time.time()
    # この例では、部分的な代入ですが、実際にはより大規模な操作に適用できます
    # 例えば、特定の条件を満たす要素全体を更新する場合など
    # ここでは、0から99までの部分を例としていますが、より効率的な方法があります。
    # より大規模なインデックス指定での代入であれば、以下のような操作の方が高速です。
    tensor[0:TEST, 0:TEST] = torch.arange(TEST).view(-1, 1) + torch.arange(TEST).view(1, -1)
    # 上記のようなブロードキャストを利用した方がはるかに高速です。
    end_time = time.time()
    print(tensor)
    print(f"スライシングとブロードキャストにかかった時間: {end_time - start_time:.4f}秒")
    
    
class Test:
    def __init__(self):
        self.n = 10
        self.powers_of_3 = torch.pow(3, torch.arange(self.n).flip(dims=[0])).long().cuda()
        
        self.tuples = torch.zeros(((15**2)*4, self.n, 2), dtype=torch.long).cuda()
        self.board = torch.zeros((15,15)).cuda()
        self.board[2][3] = 1
        self.board[5][6] = 2
        
    def get_lut_indices_by_tuples(self, board: torch.Tensor, tuples: torch.Tensor) -> torch.Tensor:
        
        # shape: (board_size**2, n)
        x_coords = tuples[:, :, 0]
        # shape: (board_size**2, n)
        y_coords = tuples[:, :, 1]
        
        # shape: (board_size**2, n)
        values = board[y_coords, x_coords]
        
        print(values.shape)
        
        # shape: (board_size**2,)
        indices = torch.sum(values * self.powers_of_3, dim=1)
        
        return indices
        
    def test2(self, boards: torch.Tensor, tuples: torch.Tensor) -> torch.Tensor:
        #self.n = 10
        
        ar_idx = torch.arange(self.boards.shape[0]).cuda()
        
        #tuples.shape = (15**2)*4, self.n, 2
        
        # shape: (15**2, n)
        x_coords = tuples[:, :, 0]
        # shape: (15**2, n)
        y_coords = tuples[:, :, 1]
        
        # boards.shape = 223, 15, 15
        # shape: (15**2, n)
        values = boards[ar_idx[:,None,None], y_coords[None,:,:], x_coords[None,:,:]]
        
        print(values.shape)
        
        # shape: (board_size**2,)
        indices = torch.sum(values[ar_idx] * self.powers_of_3, dim=1)
        
        return indices
    
    def make_boards(self):
        # self.board は (15, 15) の形状のTensorと仮定
        # board上の値は 0:空きマス, それ以外:石が置かれている とする

        # 1. 石を置ける「空きマス」の座標を直接特定する
        #    (num_empty_squares,) の形状を持つy座標とx座標のTensorが得られる
        empty_squares_xy = torch.where(self.board == 0)
        y_coords, x_coords = empty_squares_xy

        num_empty_squares = y_coords.shape[0]

        # もし空きマスがなければ、空のTensorを返して終了
        if num_empty_squares == 0:
            self.boards = torch.empty((0, 15, 15)).cuda()
            return

        # 2. 元の盤面を、空きマスの数だけ複製する
        #    expandは実際にメモリをコピーしないため、非常に高速でメモリ効率も良い
        new_boards = self.board.expand(num_empty_squares, 15, 15)

        # 3. 各盤面の対応する空きマスに、一括で石を置く（値を1加算する）
        #    [0, 1, 2, ...], [y0, y1, y2, ...], [x0, x1, x2, ...] のようにインデックスを指定し、
        #    (0, y0, x0), (1, y1, x1), ... の各要素に一気にアクセスする
        batch_indices = torch.arange(num_empty_squares).cuda()

        # new_boards は元のboardの参照なので、直接変更すると元のboardも変わってしまう可能性がある。
        # そのため、clone()で新しいメモリ領域を確保してから変更を加えるのが安全。
        self.boards = new_boards.clone()
        self.boards[batch_indices, y_coords, x_coords] += 1
    
    
    
if __name__ == "__main__":
    test = Test()
    t1 = time.time()
    
    test.make_boards()
    
    print(test.boards.shape)
    
    test.test2(test.boards,test.tuples)
    print(f"Time: {time.time()-t1:.4f}s")
    