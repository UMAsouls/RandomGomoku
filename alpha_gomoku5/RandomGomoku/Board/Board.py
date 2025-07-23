from injector import inject
from random import randint

from RandomGomoku.Interfaces import IHeadMass, IMass

from RandomGomoku.const import Stone

from RandomGomoku.Board.RandomSetter import RandomSetter

class Board():
    
    @inject
    def __init__(self, headmass: IHeadMass) -> None:
        self.__headmass: IHeadMass = headmass
        self.__board: list[list[IMass]] = []
        
        self.__width: int = -1
        self.__height: int = -1
        
        
    
    def SetStone(self, x: int, y: int, stone: Stone) -> bool:
        return self.__board[y][x].SetStone(stone)
    
    def RandomSet(self):
        """
        中央に黒石(赤)を1つ、4つの各区画に白石(青)をランダムに1つずつ配置します。
        """
        w: int = self.__width
        h: int = self.__height

        # --- 白石（青）の配置 ---
        # 4つの区画の基点となる座標を定義
        # (例: 8x8の場合、(2,2), (6,2), (2,6), (6,6) を中心としたエリア)
        quadrant_origins = [(w//4-1, h//4-1), (w - w//4-1, h//4-1), 
                            (w//4-1, h - h//4-1), (w - w//4-1, h - h//4-1)]

        # 各区画内でランダムな座標を決めるための範囲 (2x2)
        random_area_size = (2, 2)
        
        # 各区画のRandomSetterを作成
        random_setters: list[RandomSetter] = []
        for origin in quadrant_origins:
            # 2x2の範囲を持つRandomSetterをリストに追加
            random_setters.append(RandomSetter(origin[0], origin[1], 
                                            random_area_size[0], random_area_size[1]))
        
        # 各区画に1つずつ石を置く
        for setter in random_setters:
            # RandomSetterの範囲内からランダムな座標を「1つ」取得
            # (※RandomMassGet()ではなく、1つだけ座標を返すメソッドを呼び出す)
            pos = setter.RandomMassGet()
            
            # 取得した座標に白石を置く
            self.SetStone(pos[0], pos[1], Stone.WHITE)
            
        # --- 黒石（赤）の配置 ---
        # 中央の座標を計算
        center_pos = (w//2, h//2)
        
        # 中央に黒石を置く
        self.SetStone(center_pos[0], center_pos[1], Stone.BLACK)
        
        
    def MakeBoard(self, w:int, h:int):
        self.__board = self.__headmass.MakeBoard(w,h)
        
        self.__width = w
        self.__height = h
        
        self.RandomSet()
        
        
    def GetBoardInt(self) -> list[list[int]]:
        return [
            [
                i.GetStatus() for i in j
            ]
            for j in self.__board
        ]
        
        
    def PrintBoard(self) -> None:
        count1: int = 0
        count2: int = 0
        sboard: str = ""
        chg_map: dict[int, str] = {0:"🔳", 1:"🔴", 2:"🔵"}
        for i in self.__board:
            for j in i:
                s = j.GetStatus()
                sboard += f"{chg_map[s]}"
                if(s == 1): count1 += 1
                elif(s == 2): count2 += 1
            sboard += "\n"
            
        print(sboard)
        print(f"Black: {count1} White: {count2}")
  
    def copy(self): 
        return self.GetBoardInt()
        
    def IsFull(self) -> bool:
        """
        ボードが完全に石で埋まっているかをチェックします。
        埋まっている場合はTrue、そうでなければFalseを返します。
        """
        for row in self.__board:
            for mass in row:
                if mass.GetStatus() == 0:  # 0は空きマスを表す
                    return False
        return True
    
    def CheckWin(self, x: int, y: int) -> bool:
        """
        指定された位置から五目並んでいるかをチェックします。
        勝利条件を満たしていればTrue、そうでなければFalseを返します。
        """
        if x < 0 or y < 0 or x >= self.__width or y >= self.__height:
            return False
        
        stone_type = self.__board[y][x].GetStatus()
        if stone_type == 0:  # 空きマスは勝利条件を満たさない
            return False
        
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 右下対角線
            (1, -1)   # 右上対角線
        ]
        
        for dx, dy in directions:
            count = 1  # 現在位置の石をカウント
            
            # 正方向に連続する同じ石をカウント
            nx, ny = x + dx, y + dy
            while (0 <= nx < self.__width and 0 <= ny < self.__height and 
                   self.__board[ny][nx].GetStatus() == stone_type):
                count += 1
                nx += dx
                ny += dy
            
            # 負方向に連続する同じ石をカウント
            nx, ny = x - dx, y - dy
            while (0 <= nx < self.__width and 0 <= ny < self.__height and 
                   self.__board[ny][nx].GetStatus() == stone_type):
                count += 1
                nx -= dx
                ny -= dy
            
            # 5つ以上揃っていれば勝利
            if count >= 5:
                return True
        
        return False