from RandomGomoku import IBoard, GetBoard, Stone

def Test():
    board: IBoard = GetBoard(9, 9)
    board.PrintBoard()
    
    stone = Stone.BLACK
    s = "黒"
    while(True):
        board.PrintBoard()
        x = int(input(f"置くx座標({s}):"))
        if(x == -1): return
        y = int(input(f"置くy座標({s}):"))
        if(y == -1): return
        
        print("")
        if(board.SetStone(x,y,stone)): 
            board.PrintBoard()
            print(f"{s}の勝利!")
            return
            
        if(stone == Stone.WHITE):
            stone = Stone.BLACK
            s = "黒"
        else:
            stone = Stone.WHITE
            s = "白"