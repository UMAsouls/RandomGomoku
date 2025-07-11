"""
棋譜記録・保存ユーティリティ
"""

import os
import json
import datetime
import numpy as np
from typing import List, Tuple, Dict, Any


def convert_numpy_types(obj):
    """NumPy型をJSONシリアライゼーション対応の型に変換"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class GameRecorder:
    """棋譜記録・保存クラス"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.current_game_moves = []
        self.game_info = {}
        
    def start_new_game(self, board_size: int, iteration: int = None):
        """新しいゲームの記録を開始"""
        self.current_game_moves = []
        self.game_info = {
            'board_size': board_size,
            'iteration': iteration,
            'start_time': datetime.datetime.now().isoformat(),
            'winner': None,
            'total_moves': 0,
            'game_length': 0
        }
    
    def record_move(self, player: int, row: int, col: int, move_number: int = None):
        """手を記録"""
        move_info = {
            'player': player,
            'row': row,
            'col': col,
            'move_number': move_number or len(self.current_game_moves) + 1,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.current_game_moves.append(move_info)
    
    def end_game(self, winner: int, game_length: float = None):
        """ゲーム終了時の処理"""
        self.game_info['winner'] = winner
        self.game_info['total_moves'] = len(self.current_game_moves)
        self.game_info['game_length'] = game_length
        self.game_info['end_time'] = datetime.datetime.now().isoformat()
    
    def save_game_record(self, filename: str = None, format: str = 'JSON') -> str:
        """棋譜を保存"""
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            iteration = self.game_info.get('iteration', 'unknown')
            filename = f"game_record_iter{iteration}_{timestamp}"
        
        # ログディレクトリの作成
        os.makedirs(self.log_dir, exist_ok=True)
        
        if format.upper() == 'JSON':
            return self._save_as_json(filename)
        elif format.upper() == 'PGN':
            return self._save_as_pgn(filename)
        elif format.upper() == 'TEXT':
            return self._save_as_text(filename)
        elif format.upper() == 'BOARD_BY_MOVE':
            return self._save_as_board_by_move(filename)
        else:
            return self._save_as_json(filename)
    
    def _save_as_json(self, filename: str) -> str:
        """JSON形式で保存"""
        filepath = os.path.join(self.log_dir, f"{filename}.json")
        
        game_data = {
            'game_info': self.game_info,
            'moves': self.current_game_moves,
            'board_visualization': self._create_board_visualization()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
        
        return filepath
    
    def _save_as_pgn(self, filename: str) -> str:
        """PGN形式で保存（五目並べ用に改良）"""
        filepath = os.path.join(self.log_dir, f"{filename}.pgn")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # PGNヘッダー
            f.write(f'[Event "Self-Play Game"]\n')
            f.write(f'[Site "AlphaGomoku"]\n')
            f.write(f'[Date "{datetime.datetime.now().strftime("%Y.%m.%d")}"]\n')
            f.write(f'[Round "{self.game_info.get("iteration", "?")}"]\n')
            f.write(f'[White "AlphaGomoku"]\n')
            f.write(f'[Black "AlphaGomoku"]\n')
            
            winner = self.game_info.get('winner', 0)
            if winner == 1:
                result = '1-0'
            elif winner == 2:
                result = '0-1'
            else:
                result = '1/2-1/2'
            f.write(f'[Result "{result}"]\n')
            f.write(f'[BoardSize "{self.game_info.get("board_size", 8)}"]\n')
            f.write(f'[TotalMoves "{self.game_info.get("total_moves", 0)}"]\n')
            f.write('\n')
            
            # 棋譜
            for i, move in enumerate(self.current_game_moves):
                if i % 2 == 0:
                    f.write(f'{i//2 + 1}. ')
                
                # 座標を棋譜表記に変換（例: a1, b2, etc.）
                col_letter = chr(ord('a') + move['col'])
                row_number = move['row'] + 1
                f.write(f'{col_letter}{row_number} ')
                
                if (i + 1) % 10 == 0:  # 10手ごとに改行
                    f.write('\n')
            
            f.write(f' {result}\n')
        
        return filepath
    
    def _save_as_text(self, filename: str) -> str:
        """テキスト形式で保存（一手ずつ盤面表示）"""
        filepath = os.path.join(self.log_dir, f"{filename}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("AlphaGomoku 自己対戦棋譜\n")
            f.write("=" * 50 + "\n\n")
            
            # ゲーム情報
            f.write("ゲーム情報:\n")
            f.write(f"  盤面サイズ: {self.game_info.get('board_size', 8)}x{self.game_info.get('board_size', 8)}\n")
            f.write(f"  イテレーション: {self.game_info.get('iteration', 'Unknown')}\n")
            f.write(f"  開始時刻: {self.game_info.get('start_time', 'Unknown')}\n")
            f.write(f"  終了時刻: {self.game_info.get('end_time', 'Unknown')}\n")
            f.write(f"  総手数: {self.game_info.get('total_moves', 0)}\n")
            f.write(f"  勝者: プレイヤー {self.game_info.get('winner', 'Unknown')}\n")
            if self.game_info.get('game_length'):
                f.write(f"  対局時間: {self.game_info.get('game_length', 0):.2f}秒\n")
            f.write("\n")
            
            # 一手ずつ棋譜と盤面を表示
            f.write("棋譜（一手ずつ盤面表示）:\n")
            f.write("=" * 50 + "\n\n")
            
            # 初期盤面
            f.write("初期盤面\n")
            f.write("=" * 20 + "\n")
            board_size = self.game_info.get('board_size', 8)
            initial_board = self._create_empty_board(board_size)
            self._write_board_to_file(f, initial_board)
            f.write("\n")
            
            # 各手の後の盤面を表示
            for i, move in enumerate(self.current_game_moves):
                player_name = "先手" if move['player'] == 1 else "後手"
                f.write(f"{i+1:2d}手目: {player_name} ({move['row']:2d}, {move['col']:2d})\n")
                f.write("=" * 20 + "\n")
                
                # この手までの盤面を作成
                board_at_move = self._create_board_at_move(i)
                self._write_board_to_file(f, board_at_move)
                f.write("\n")
        
        return filepath
    
    def _create_board_visualization(self) -> List[List[str]]:
        """盤面の可視化を作成"""
        board_size = self.game_info.get('board_size', 8)
        board = [['.' for _ in range(board_size)] for _ in range(board_size)]
        
        for move in self.current_game_moves:
            row, col = move['row'], move['col']
            player = move['player']
            if 0 <= row < board_size and 0 <= col < board_size:
                board[row][col] = '●' if player == 1 else '○'
        
        return board
    
    def _create_empty_board(self, board_size: int) -> List[List[str]]:
        """空の盤面を作成"""
        return [['.' for _ in range(board_size)] for _ in range(board_size)]
    
    def _create_board_at_move(self, move_index: int) -> List[List[str]]:
        """指定した手番までの盤面を作成"""
        board_size = self.game_info.get('board_size', 8)
        board = [['.' for _ in range(board_size)] for _ in range(board_size)]
        
        # move_indexまでの手を盤面に配置
        for i in range(move_index + 1):
            if i < len(self.current_game_moves):
                move = self.current_game_moves[i]
                row, col = move['row'], move['col']
                player = move['player']
                if 0 <= row < board_size and 0 <= col < board_size:
                    board[row][col] = '●' if player == 1 else '○'
        
        return board
    
    def _write_board_to_file(self, file_handle, board: List[List[str]]):
        """盤面をファイルに書き込み"""
        for row in board:
            file_handle.write("  " + " ".join(row) + "\n")
    
    def get_game_summary(self) -> Dict[str, Any]:
        """ゲームの要約情報を取得"""
        return {
            'total_moves': len(self.current_game_moves),
            'winner': self.game_info.get('winner'),
            'game_length': self.game_info.get('game_length'),
            'board_size': self.game_info.get('board_size'),
            'iteration': self.game_info.get('iteration')
        }


def create_game_recorder(log_dir: str) -> GameRecorder:
    """GameRecorderインスタンスを作成"""
    return GameRecorder(log_dir)
