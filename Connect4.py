import numpy as np
import pygame
from scipy import signal
import threading

# version
version = "Connect4-v0.1"

# Initialize Pygame
pygame.init()

# Constants for players
AI_PLAYER = 1
HUMAN_PLAYER = -1

# Search depth for the AI
DEPTH = 4

# Screen dimensions
ROW_COUNT = 8
COLUMN_COUNT = 10
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# Create the screen
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Connect 4")

# Win conditions for convolution
WIN_CONDS = [
    # Vertical win condition
    np.array([[1], [1], [1], [1]]),
    
    # Horizontal win condition
    np.array([[1, 1, 1, 1]]),
    
    # Diagonal win condition (top-left to bottom-right)
    np.array([[1, 0, 0, 0], 
              [0, 1, 0, 0], 
              [0, 0, 1, 0], 
              [0, 0, 0, 1]]),
    
    # Diagonal win condition (bottom-left to top-right)
    np.array([[0, 0, 0, 1], 
              [0, 0, 1, 0], 
              [0, 1, 0, 0], 
              [1, 0, 0, 0]])
]

# Global counter for nodes evaluated
nodes_evaluated = 0

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int((r + 1) * SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if board[r][c] == AI_PLAYER:
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), int(height - (ROW_COUNT - r - 1) * SQUARESIZE - SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == HUMAN_PLAYER:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), int(height - (ROW_COUNT - r - 1) * SQUARESIZE - SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

def display_board(board):
    print("  ".join(map(str, range(COLUMN_COUNT))))
    for r in range(ROW_COUNT):
        print("  ".join(['X' if cell == HUMAN_PLAYER else 'O' if cell == AI_PLAYER else '.' for cell in board[r]]))

def place_piece(board, col, piece):
    for r in range(ROW_COUNT - 1, -1, -1):  
        if board[r][col] == 0:
            board[r][col] = piece
            return board
    return board

def is_win(board, piece):
    piece_board = (board == piece).astype(int)
    for cond in WIN_CONDS:
        conv = signal.convolve2d(piece_board, cond, mode='valid')
        if np.any(conv == 4):
            return True
    return False

def is_draw(board):
    return not is_win(board, AI_PLAYER) and not is_win(board, HUMAN_PLAYER) and not np.any(board == 0)

def valid_moves(board):
    return [idx for idx, value in enumerate(board[0]) if value == 0]

def alpha_beta(node, depth, alpha, beta, max_depth=DEPTH):
    global nodes_evaluated
    nodes_evaluated += 1

    if depth == 0 or node.is_terminal():
        return score_board(node.board, depth)
    if node.is_maximizing_player():
        value = -np.inf
        for child, _ in node.get_children():
            value = max(value, alpha_beta(child, depth - 1, alpha, beta, max_depth))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.inf
        for child, _ in node.get_children():
            value = min(value, alpha_beta(child, depth - 1, alpha, beta, max_depth))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def score_board(board, depth):
    player_score = score_player_board(board)
    score = player_score - depth / 1000
    if is_win(board, AI_PLAYER):
        score = 4
    if is_win(board, HUMAN_PLAYER):
        score = -4
    return score

def score_player_board(board):
    convs = [signal.convolve2d((board == AI_PLAYER).astype(int), cond, mode='valid') for cond in WIN_CONDS]
    flat_convs = np.array([i for conv in convs for row in conv for i in row])
    best = np.max(flat_convs)
    threes = np.sum(flat_convs == 3)
    twos = np.sum(flat_convs == 2)
    score = best + threes / 10 + twos / 100
    score = min(4, score)
    return score

class PlayerNode:
    def __init__(self, board):
        self.board = board
        self.player = AI_PLAYER

    def get_children(self):
        moves = valid_moves(self.board)
        children = [(EnemyNode(place_piece(np.copy(self.board), move, self.player)), move) for move in moves]
        return children

    def is_terminal(self):
        return is_win(self.board, self.player) or is_draw(self.board)

    def is_maximizing_player(self):
        return True

class EnemyNode:
    def __init__(self, board):
        self.board = board
        self.player = HUMAN_PLAYER

    def get_children(self):
        moves = valid_moves(self.board)
        children = [(PlayerNode(place_piece(np.copy(self.board), move, self.player)), move) for move in moves]
        return children

    def is_terminal(self):
        return is_win(self.board, self.player) or is_draw(self.board)

    def is_maximizing_player(self):
        return False

def play_move(board):
    global nodes_evaluated
    nodes_evaluated = 0

    moves = valid_moves(board)
    next_boards = [place_piece(np.copy(board), move, AI_PLAYER) for move in moves]
    next_nodes = [EnemyNode(board) for board in next_boards]
    scores = [process_node(node) for node in next_nodes]
    moves_scores = sorted(zip(moves, scores), key=lambda x: x[1])
    best_move = moves_scores[-1][0]

    # print(f"AI considering moves: {moves}")
    print(f"Move scores: {scores}")
    print(f"Total number of nodes considered: {nodes_evaluated}")
    print(f"Depth: {DEPTH}")
    print(f"Best move: {best_move}")

    return best_move

def process_node(node):
    return alpha_beta(node, DEPTH, -np.inf, np.inf)

def ai_turn(board, turn_event):
    col = play_move(board)
    print(f"AI move: column {col}")
    if board[0][col] == 0:
        place_piece(board, col, AI_PLAYER)
        turn_event.set()

def display_message(message):
    font = pygame.font.SysFont("monospace", 75)
    label = font.render(message, 1, WHITE)
    screen.blit(label, (40, 10))
    pygame.display.update()
    pygame.time.wait(3000)

def main():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    game_over = False
    turn = 0

    draw_board(board)
    display_board(board)
    pygame.display.update()

    ai_thread = None
    turn_event = threading.Event()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if turn == 0:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                else:
                    pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN and turn == 0:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                col = int(posx // SQUARESIZE)
                print(f"Player move: column {col}")

                if board[0][col] == 0:
                    place_piece(board, col, HUMAN_PLAYER)
                    draw_board(board)
                    display_board(board)

                    if is_win(board, HUMAN_PLAYER):
                        print("Player wins!")
                        display_message("Player wins!")
                        game_over = True
                    elif is_draw(board):
                        print("Game is a draw!")
                        display_message("Game is a draw!")
                        game_over = True
                    else:
                        turn = 1
                        turn_event.clear()

        if not game_over and turn == 1 and not ai_thread:
            ai_thread = threading.Thread(target=ai_turn, args=(board, turn_event))
            ai_thread.start()

        if turn_event.is_set():
            display_message("Thinking...")
            ai_thread.join()
            ai_thread = None
            turn_event.clear()
            draw_board(board)
            display_board(board)
            if is_win(board, AI_PLAYER):
                print("AI wins!")
                display_message("AI wins!")
                game_over = True
            elif is_draw(board):
                print("Game is a draw!")
                display_message("Game is a draw!")
                game_over = True
            else:
                turn = 0

        if game_over:
            pygame.time.wait(30000)
            pygame.quit()
            return

if __name__ == "__main__":
    main()
    print("Game over.")


