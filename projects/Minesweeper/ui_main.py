import pygame

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Minesweeper")
        self.clock = pygame.time.Clock()
    
    
    def new(self):
        pass
    
    
    def run(self):
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.draw()
    
    def draw(self):
        self.screen.fill(BG_COLOR)
        pygame.display.flip()
        
        
    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit(0)
                
game = Game()
while True:
    game.new()
    game.run()
                