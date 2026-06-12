import random
import pygame 
from settings import *


class Tile () :
    def __init__ (self , x , y , image , type , revealed = False , flagged = False) :
        self.x , self.y = x * TILE_SIZE , y * TILE_SIZE
        self.image = image
        self.type = type
        self.revealed = revealed
        self.flagged = flagged
        
        
    def draw (self , board_surface) :
        if not self.flagged and self.revealed :
            board_surface.blit(self.image , (self.x , self.y))
        elif self.flagged and not self.revealed :
            board_surface.blit(tile_flag , (self.x , self.y))
        elif not self.revealed : 
            board_surface.blit(tile_unknown , (self.x , self.y))
        
        
    def __repr__ (self) :
        return self.type


class Board () :
    def __init__ (self) :
        self.board_surface = pygame.Surface((WIDTH, HEIGHT))
        self.board_list = [[Tile(ROWS , COLS , tile_empty , ".") for ROWS in range(ROWS)] for COLS in range(COLS)]
        self.place_mines()
        self.place_clues()
        self.dug = []
        
    
    def place_mines (self) :
        for _ in range(AMOUNT_MINES) :
            while True :
                x = random.randint(0 , ROWS - 1)
                y = random.randint(0 , COLS - 1)
                
                if self.board_list[y][x].type == "." :
                    self.board_list[y][x].image = tile_mine
                    self.board_list[y][x].type = "M"
                    break 
                
    def place_clues(self):
        for y in range(COLS):
            for x in range(ROWS):
                if self.board_list[y][x].type != "M":
                    total_mines = self.check_neighbors(x, y)
                    if total_mines > 0:
                        self.board_list[y][x].image = title_number[total_mines - 1]
                        self.board_list[y][x].type = "C"
    
    @staticmethod
    def is_inside(x , y):
        return 0 <= x < COLS and 0 <= y < ROWS

    def check_neighbors(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= x + i < COLS and 0 <= y + j < ROWS:
                    if self.board_list[y + j][x + i].type == "M":
                        count += 1
        return count
    
    
    
    
    def draw (self , screen) :
        for row in self.board_list :
            for tile in row :
                tile.draw(self.board_surface)
        screen.blit(self.board_surface , (0 , 0)) 
    
    
    def dig(self , x , y) :
        self.dug.append((x , y))
        if self.board_list[y][x].type == "M" :
            self.board_list[y][x].revealed = True
            self.board_list[y][x].image = tile_exploded
            return False
        
        elif self.board_list[y][x].type == "C" :
            self.board_list[y][x].revealed = True
            return True
        
        
        self.board_list[y][x].revealed = True
        for i in range(max(0 , x - 1) , min(COLS - 1 , x + 1)) :
            for j in range(max(0 , y - 1) , min(ROWS - 1 , y + 1)) :
                if (i , j) not in self.dug :
                    self.dig(i , j)
        return True
        
    def display_board (self) :
        for row in self.board_list :
            print(row)