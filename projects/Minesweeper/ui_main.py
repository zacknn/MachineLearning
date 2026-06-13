import pygame
from settings import *
from sprites import *
from button import Button
from ai_player import AIPlayer

class Game:
    def __init__(self):
        pygame.init()
        # Update window size to include buttons
        self.window_width = WIDTH + BUTTON_WIDTH + 50
        self.screen = pygame.display.set_mode((self.window_width, HEIGHT))
        pygame.display.set_caption("Minesweeper with AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Create buttons
        self.play_button = Button(
            BUTTON_X, PLAY_BUTTON_Y, 
            BUTTON_WIDTH, BUTTON_HEIGHT, 
            "PLAY", BUTTON_COLOR, BUTTON_HOVER_COLOR, BUTTON_TEXT_COLOR
        )
        
        self.ai_button = Button(
            BUTTON_X, AI_BUTTON_Y, 
            BUTTON_WIDTH, BUTTON_HEIGHT, 
            "AI PLAY", BUTTON_COLOR, BUTTON_HOVER_COLOR, BUTTON_TEXT_COLOR
        )
        
        self.board = None
        self.ai_player = None
        self.game_mode = None  # "human" or "ai"
        self.ai_move_timer = 0  # For controlling AI speed
        
    def new(self, mode="human"):
        """Start a new game in specified mode"""
        self.board = Board()
        self.game_mode = mode
        self.board.display_board()
        
        if mode == "ai":
            self.ai_player = AIPlayer(self)
            self.ai_move_timer = pygame.time.get_ticks()
    
    def run(self):
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.update()  # New method for AI updates
            self.draw()
        else:
            self.end_screen()
    
    def update(self):
        """Handle AI moves"""
        if self.game_mode == "ai" and self.playing:
            # Make AI move every 0.5 seconds (adjust for speed)
            current_time = pygame.time.get_ticks()
            if current_time - self.ai_move_timer > 500:  # 500ms delay
                self.ai_move_timer = current_time
                if not self.ai_player.make_move():
                    self.playing = False
    
    def check_win(self):
        for row in self.board.board_list:
            for tile in row:
                if tile.type != "M" and not tile.revealed:
                    return False
        return True
    
    def draw(self):
        # Draw background
        self.screen.fill(BG_COLOR)
        
        # Draw game board (centered, with offset for buttons)
        if self.board:
            self.board.draw(self.screen)
        
        # Draw buttons
        self.play_button.draw(self.screen, self.font)
        self.ai_button.draw(self.screen, self.font)
        
        # Draw current mode text
        mode_text = f"Mode: {'AI' if self.game_mode == 'ai' else 'You'}"
        text_surface = self.font.render(mode_text, True, WHITE)
        self.screen.blit(text_surface, (BUTTON_X, AI_BUTTON_Y + BUTTON_HEIGHT + 20))
        
        pygame.display.flip()
    
    def handle_click(self, x, y, button):
        """Handle mouse clicks on the grid"""
        # Check if click is within grid bounds
        if 0 <= x < ROWS and 0 <= y < COLS:
            if button == 1:  # Left click
                if not self.board.board_list[y][x].flagged:
                    if not self.board.dig(x, y):
                        # Hit a mine - game over
                        for row in self.board.board_list:
                            for tile in row:
                                if tile.flagged and tile.type != "M":
                                    tile.flagged = False
                                    tile.revealed = True
                                    tile.image = tile_not_mine
                                elif not tile.flagged and tile.type == "M":
                                    tile.revealed = True
                        self.playing = False
            
            elif button == 3:  # Right click
                if not self.board.board_list[y][x].revealed:
                    self.board.board_list[y][x].flagged = not self.board.board_list[y][x].flagged
            
            # Check win after move
            if self.check_win():
                self.win = True
                self.playing = False
                for row in self.board.board_list:
                    for tile in row:
                        if not tile.revealed:
                            tile.flagged = True
    
    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit(0)
            
            # Handle button clicks
            if self.play_button.handle_event(event):
                self.new("human")
            
            if self.ai_button.handle_event(event):
                self.new("ai")
            
            # Handle mouse clicks on grid (only in human mode)
            if event.type == pygame.MOUSEBUTTONDOWN and self.game_mode == "human":
                mx, my = pygame.mouse.get_pos()
                # Only handle clicks on the grid area
                if mx < WIDTH and my < HEIGHT:
                    x = mx // TILE_SIZE
                    y = my // TILE_SIZE
                    self.handle_click(x, y, event.button)
    
    def end_screen(self):
        """Show end screen with win/loss message"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit(0)
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            
            # Draw end screen
            self.screen.fill(BG_COLOR)
            
            # Show result
            if hasattr(self, 'win') and self.win:
                text = "YOU WIN!" if self.game_mode == "You" else "AI WINS!"
            else:
                text = "GAME OVER!" if self.game_mode == "You" else "AI LOST!"
            
            text_surface = self.font.render(text, True, WHITE)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)
            
            # Instructions
            inst_surface = self.font.render("Click to play again", True, LIGHTGRAY)
            inst_rect = inst_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            self.screen.blit(inst_surface, inst_rect)
            
            pygame.display.flip()
            self.clock.tick(FPS)

# Main game loop
game = Game()
while True:
    game.new("You")  # Start in human mode
    game.run()