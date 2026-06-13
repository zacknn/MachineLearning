import pygame
import os
# colorss
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGRAY = (40, 40, 40)
LIGHTGRAY = (100, 100, 100)
GREEN = (0, 255, 0)
DARKGREEN = (0, 200, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BG_COLOR = DARKGRAY

# game settings
TILE_SIZE = 32
ROWS = 15
COLS = 15
FPS = 60
WIDTH = COLS * TILE_SIZE
HEIGHT = ROWS * TILE_SIZE
AMOUNT_MINES = 20


#Buttons
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
BUTTON_X = WIDTH + 20  # Place buttons to the right of the grid
PLAY_BUTTON_Y = 100
AI_BUTTON_Y = 200
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = WHITE

WINDOW_WIDTH = WIDTH + BUTTON_WIDTH + 40  # Add padding

title_number = []
for i in range(1, 9):
    title_number.append(pygame.transform.scale(pygame.image.load(os.path.join('assets', f'Tile{i}.png')), (TILE_SIZE, TILE_SIZE)))

tile_empty = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'TileEmpty.png')), (TILE_SIZE, TILE_SIZE))
tile_exploded = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'TileExploded.png')), (TILE_SIZE, TILE_SIZE))
tile_flag = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'TileFlag.png')), (TILE_SIZE, TILE_SIZE))
tile_mine = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'TileMine.png')), (TILE_SIZE, TILE_SIZE))
tile_not_mine = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'TileNotMine.png')), (TILE_SIZE, TILE_SIZE))
tile_unknown = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'TileUnknown.png')), (TILE_SIZE, TILE_SIZE))