import pygame
from settings import *
from sprites import *


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Minesweeper")
        self.clock = pygame.time.Clock()
        self.board = Board()


    def new(self):
        self.board = Board()
        self.board.display_board()
    
    
    def run(self):
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.draw()
            
        else : 
            self.end_screen()
            
    def check_win(self):
        for row in self.board.board_list:
            for tile in row:
                if tile.type != "M" and not tile.revealed:
                    return False
        return True
    
    def draw(self):
        self.screen.fill(BG_COLOR)
        self.board.draw(self.screen)
        pygame.display.flip()
        
        
    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit(0)
                
            if event.type == pygame.MOUSEBUTTONDOWN :
                mx , my = pygame.mouse.get_pos()
                x = mx // TILE_SIZE
                y = my // TILE_SIZE
            
                if event.button == 1 :
                    if not self.board.board_list[y][x].flagged :
                        if not self.board.dig(x , y) :
                            for row in self.board.board_list : 
                                for tile in row :
                                    if tile.flagged and tile.type != "M" :
                                        tile.flagged = False
                                        tile.revealed = True
                                        tile.image = tile_not_mine
                                    elif not tile.flagged and tile.type == "M" :
                                        tile.revealed = True
                            
                            self.playing = False
                        
                if event.button == 3 :
                    if not self.board.board_list[y][x].revealed :
                        self.board.board_list[y][x].flagged = not self.board.board_list[y][x].flagged
                        
                        
                if self.check_win() :
                    self.win = True
                    self.playing = False
                    for row in self.board.board_list :
                        for tile in row :
                            if not tile.revealed : 
                                tile.flagged = True
                                
    def end_screen(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit(0)
                    
                if event.type == pygame.MOUSEBUTTONDOWN :
                    return 
                
game = Game()
while True:
    game.new()
    game.run()
                