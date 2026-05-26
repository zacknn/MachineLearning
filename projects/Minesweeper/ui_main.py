import pygame
import sys

pygame.init()

# Settings
ROWS, COLS = 8, 8
CELL_SIZE = 50
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
Mines = 8
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8x8 Grid")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Create 8x8 grid (matrix) - you can store data here later
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    window.fill(WHITE)

    # Draw the grid
    for row in range(ROWS):
        for col in range(COLS):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            
            color = GRAY
        
            
            pygame.draw.rect(window, color, (x, y, CELL_SIZE, CELL_SIZE))
            
            # Optional: Draw grid lines
            pygame.draw.rect(window, WHITE, (x, y, CELL_SIZE, CELL_SIZE), 1)

    pygame.display.flip()

pygame.quit()
sys.exit()