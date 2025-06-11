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
        w:int = self.__width
        h: int = self.__height
        
        white_rand_size = (w*2//5, h*2//5)
        black_rand_size = (w*1//5, h*1//5)
        
        white_rand1: RandomSetter = RandomSetter(0, 0, white_rand_size[0], white_rand_size[1])
        white_rand2: RandomSetter = RandomSetter(w-white_rand_size[0], 0, white_rand_size[0], white_rand_size[1])
        white_rand3: RandomSetter = RandomSetter(0, h-white_rand_size[1], white_rand_size[0], white_rand_size[1])
        white_rand4: RandomSetter = RandomSetter(w-white_rand_size[0], h-white_rand_size[1], white_rand_size[0], white_rand_size[1])
        
        white_rands: list[RandomSetter] = [
            white_rand1,white_rand2,white_rand3,white_rand4
        ]
        
        black_rand1: RandomSetter = RandomSetter(black_rand_size[0], 0, black_rand_size[0], h)
        black_rand2: RandomSetter = RandomSetter(0, black_rand_size[1], w, black_rand_size[1])
        
        black_rands: list[RandomSetter] = [
            black_rand1, black_rand2
        ]
        
        for i in white_rands:
            pos = i.RandomMassGet()
            self.SetStone(pos[0], pos[1], Stone.WHITE)
            
        
        bpos1 = black_rand1.RandomMassGet()
        bpos2 = black_rand2.RandomMassGet()
        
        if(randint(0,1) == 0): bpos = bpos1
        else: bpos = bpos2
        
        self.SetStone(bpos[0], bpos[1], Stone.BLACK)
        
        
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