import numpy as np
import pygame
import sys
from alpha_gomoku import AlphaGomokuAgent
import GomokuEnv

# Initialize Pygame
pygame.init()
BOARD_SIZE = 19
CELL_SIZE = 30
MARGIN = 20
WIDTH = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AlphaGomoku")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (210, 180, 140)

def draw_board(board):
    """Draw the Gomoku board"""
    # Draw the wooden background
    SCREEN.fill(BROWN)
    
    # Draw the grid lines
    for i in range(BOARD_SIZE):
        # Vertical lines
        pygame.draw.line(SCREEN, BLACK, 
                        (MARGIN + i * CELL_SIZE, MARGIN), 
                        (MARGIN + i * CELL_SIZE, HEIGHT - MARGIN), 1)
        # Horizontal lines
        pygame.draw.line(SCREEN, BLACK, 
                        (MARGIN, MARGIN + i * CELL_SIZE), 
                        (WIDTH - MARGIN, MARGIN + i * CELL_SIZE), 1)
    
    # Draw the stones
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:  # Black stone
                pygame.draw.circle(SCREEN, BLACK, 
                                  (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE), 
                                  CELL_SIZE // 2 - 2)
            elif board[i][j] == 2:  # White stone
                pygame.draw.circle(SCREEN, WHITE, 
                                  (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE), 
                                  CELL_SIZE // 2 - 2)

def play_game():
    """Play a game against the AI"""
    # Load the trained agent
    agent = AlphaGomokuAgent(board_size=BOARD_SIZE)
    try:
        agent.load_model("models/alpha_gomoku_final.pt")
    except:
        print("No trained model found. Using an untrained agent.")
    
    # Create the environment
    env = GomokuEnv.GomokuEnv()
    state = env.board.copy()
    done = False
    
    # Game loop
    while not done:
        # Draw the board
        draw_board(state)
        pygame.display.flip()
        
        current_player = env.current_player
        
        if current_player == 1:  # Human plays black
            # Wait for player input
            move_made = False
            while not move_made:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Get the mouse position
                        x, y = pygame.mouse.get_pos()
                        # Convert to board coordinates
                        col = round((x - MARGIN) / CELL_SIZE)
                        row = round((y - MARGIN) / CELL_SIZE)
                        
                        # Check if the move is valid
                        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and state[row][col] == 0:
                            action = (col, row)
                            next_state, reward, done, _ = env.step(action)
                            state = next_state.copy()
                            move_made = True
        else:  # AI plays white
            # Get AI move
            action = agent.get_action(state, current_player)
            next_state, reward, done, _ = env.step(action)
            state = next_state.copy()
        
        # Draw the updated board
        draw_board(state)
        pygame.display.flip()
        
        # Check if game is over
        if done:
            winner = env.winner
            if winner == 0:
                result = "Draw!"
            elif winner == 1:
                result = "You win!"
            else:
                result = "AI wins!"
            
            # Display the result
            font = pygame.font.Font(None, 36)
            text = font.render(result, True, BLACK)
            text_rect = text.get_rect(center=(WIDTH//2, HEIGHT//2))
            SCREEN.blit(text, text_rect)
            pygame.display.flip()
            
            # Wait for a few seconds before quitting
            pygame.time.wait(3000)
            
            return

if __name__ == "__main__":
    play_game()
