import sys
from collections import deque

board = [["-" for _ in range(9)] for _ in range(9)]
black = True
active = "X"

def get_lines(coordinate, active):
    global scores
    row = coordinate // 9
    column = coordinate % 9
    
    lines = []
    if active == "X":
        passive = "O"
    else:
        passive = "X"
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        line = deque("T")
        for i in range(1, 9):
            tmp_row = row + i * dr
            tmp_column = column + i * dc
            
            if not (0 <= tmp_row < 9 and 0 <= tmp_column < 9):
                break
            stone = board[tmp_row][tmp_column]
            if stone == passive:
                break
            line.append(stone)
        for i in range(1, 9):
            tmp_row = row - i * dr
            tmp_column = column - i * dc

            if not (0 <= tmp_row < 9 and 0 <= tmp_column < 9):
                break
            stone = board[tmp_row][tmp_column]
            if stone == passive:
                break
            line.appendleft(stone)
        lines.append(line)
        scores[coordinate] += len(line)

    return lines

def windowing(line, active):
    line = list(line)
    segments = []
    if len(line) < 5:
        segments.append(line + [-1])
    for i in range(len(line) - 4):
        segment = line[i:i + 5]
        if "T" not in segment:
            continue
        segment[segment.index("T")] = active
        segments.append(segment + [len(line) - 5])
    return segments

def calc_oneline_score(line, active, me):
    line[line.index("T")] = active
    tmp_score = 0
    if len(line) < 5:
        return tmp_score
    
    line = "".join(line)
    # tmp_score += len(line)
    # achievement = []
    #五
    if "".join([active, active, active, active, active]) in line:
        if active == me:
            tmp_score += 10 ** 68
        else:
            tmp_score += 10 ** 66
    #両端空いてる連続4
    if "".join(["-", active, active, active, active, "-"]) in line:
        if active == me:
            tmp_score += 10 ** 64
        else:
            tmp_score += 10 ** 62
    #片方空いてる連続4
    if "".join(["-", active, active, active, active]) in line or "".join([active, active, active, active, "-"]) in line:
        if active == me:
            tmp_score += 10 ** 60
        else:
            tmp_score += 10 ** 58
    #両端空いてる13
    if "".join(["-", active, "-", active, active, active, "-"]) in line or "".join(["-", active, active, active, "-", active, "-"]) in line:
        if active == me:
            tmp_score += 10 ** 56
        else:
            tmp_score += 10 ** 54
    #1側だけ止められてる13
    if "".join([active, "-", active, active, active, "-"]) in line or "".join(["-", active, active, active, "-", active]) in line:
        if active == me:
            tmp_score += 10 ** 52
        else:
            tmp_score += 10 ** 50
    #両側空いてる連続3
    if "".join(["-", active, active, active, "-"]) in line:
        if active == me:
            tmp_score += 10 ** 48
        else:
            tmp_score += 10 ** 46
    #両側止められてる13
    if "".join([active, "-", active, active, active]) in line or "".join([active, active, active, "-", active]) in line:
        if active == me:
            tmp_score += 10 ** 44
        else:
            tmp_score += 10 ** 42
    #2-2
    if "".join([active, active, "-", active, active]) in line:
        if active == me:
            tmp_score += 10 ** 40
        else:
            tmp_score += 10 ** 38
    #片側3
    if "".join([active, active, active, "-", "-"]) in line or "".join(["-", "-", active, active, active]) in line:
        if active == me:
            tmp_score += 10 ** 36
        else:
            tmp_score += 10 ** 34
    #飛び3
    if "".join(["-", active, "-", active, active]) in line or "".join([active, "-", active, active, "-"]) in line or "".join([active, active, "-", active, "-"]) in line or "".join(["-", active, active, "-", active]) in line:
        if active == me:
            tmp_score += 10 ** 32
        else:
            tmp_score += 10 ** 30
    #1-1-1
    if "".join([active, "-", active, "-", active]) in line:
        if active == me:
            tmp_score += 10 ** 28
        else:
            tmp_score += 10 ** 26
    #2
    if "".join([active, active, "-", "-", "-"]) in line or "".join(["-", active, active, "-", "-"]) in line or "".join(["-", "-", active, active, "-"]) in line or "".join(["-", "-", "-", active, active]) in line:
        if active == me:
            tmp_score += 10 ** 6
        else:
            tmp_score += 10 ** 4
    return tmp_score

def judge_turn(board):
    cnt_black = 0
    cnt_white = 0
    for i in range(81):
        if board[i // 9][i % 9] == "X":
            cnt_black += 1
        elif board[i // 9][i % 9] == "O":
            cnt_white += 1
    if (cnt_white - cnt_black) % 2 == 0:
        return "O"
    else:
        return "X"

while True:
    cmd = sys.stdin.readline().strip().split()
    if cmd[0] == "quit":
        break
    elif cmd[0] == "pos":
        cnt_go = 0
        black = None
        for i in range(81):
            board[i // 9][i % 9] = cmd[1][i]
        # if cmd[2] == "O":
        #     black = False
        #     active = "O"
        default_scores = [0 for _ in range(81)]
        for i in range(81):
            if board[i // 9][i % 9] != "-":
                default_scores[i] -= 10 ** 100
        # if black:
        #     for i in range(81):
        #         default_scores[i] += 4 - max(abs(i // 9 - 4), abs(i % 9 - 4))
        # else:
        #     cnt_white = 0
        #     for i in range(81):
        #         if board[i // 9][i % 9] == "O" and cnt_white == 0:
        #             default_scores[i + 1] += 4
        #             default_scores[i + 9] += 4
        #             cnt_white += 1
        #         elif board[i // 9][i % 9] == "O" and cnt_white == 1:
        #             default_scores[i - 1] += 4
        #             default_scores[i + 9] += 4
        #             cnt_white += 1
        #         elif board[i // 9][i % 9] == "O" and cnt_white == 2:
        #             default_scores[i + 1] += 4
        #             default_scores[i - 9] += 4
        #             cnt_white += 1
        #         elif board[i // 9][i % 9] == "O" and cnt_white == 3:
        #             default_scores[i - 1] += 4
        #             default_scores[i - 9] += 4
        #             cnt_white += 1

    elif cmd[0] == "move":
        move = int(cmd[1])
        if black == None:
            if judge_turn(board) == "X":
                black = False
                active = "O"
            else:
                black = True
                active = "X"
        if black:
            board[move // 9][move % 9] = "O"
        else:
            board[move // 9][move % 9] = "X"
        default_scores[move] -= 10 ** 100
    elif cmd[0] == "go":
        if black == None:
            if judge_turn(board) == "X":
                black = True
                active = "X"
            else:
                black = False
                active = "O"
        if cnt_go == 0:
            if black:
                for i in range(81):
                    default_scores[i] += 4 - max(abs(i // 9 - 4), abs(i % 9 - 4))
            else:
                cnt_white = 0
                for i in range(81):
                    if board[i // 9][i % 9] == "O" and cnt_white == 0:
                        default_scores[i + 1] += 4
                        default_scores[i + 9] += 4
                        cnt_white += 1
                    elif board[i // 9][i % 9] == "O" and cnt_white == 1:
                        default_scores[i - 1] += 4
                        default_scores[i + 9] += 4
                        cnt_white += 1
                    elif board[i // 9][i % 9] == "O" and cnt_white == 2:
                        default_scores[i + 1] += 4
                        default_scores[i - 9] += 4
                        cnt_white += 1
                    elif board[i // 9][i % 9] == "O" and cnt_white == 3:
                        default_scores[i - 1] += 4
                        default_scores[i - 9] += 4
                        cnt_white += 1
            cnt_go += 1

        scores = default_scores[:]
        for i in range(81):
            for j in get_lines(i, "X"):
                scores[i] += calc_oneline_score(list(j), "X", active)
            for j in get_lines(i, "O"):
                scores[i] += calc_oneline_score(list(j), "O", active)
        maxn = -1
        move = -1
        for i in range(81):
            if maxn < scores[i]:
                maxn = scores[i]
                move = i
        print("move", move)
        sys.stdout.flush()
        board[move // 9][move % 9] = active
        default_scores[move] -= 10 ** 100
    elif cmd[0] == "test":
        tmp_score = 0
        for i in get_lines(int(cmd[1]), "X"):
            tmp_score += calc_oneline_score(list(i), "X", "X")
        for i in get_lines(int(cmd[1]), "O"):
            tmp_score += calc_oneline_score(list(i), "O" ,"X")
        print(tmp_score)          

    for i in board:
        print(*i)
