import random
from collections import deque
from settings import *


class AIPlayer:
    def __init__(self, game):
        self.game = game
        self.safe_cells = []  # Known safe cells to click
        self.mine_cells = []  # Known mine cells to flag
        
    def make_move(self):
        """Make one move - returns True if move was made, False if game over"""
        
        # First, update knowledge based on current board
        self.update_knowledge()
        
        # Strategy 1: Click known safe cells
        if self.safe_cells:
            x, y = self.safe_cells.pop(0)
            return self.left_click(x, y)
        
        # Strategy 2: Flag known mines
        if self.mine_cells:
            x, y = self.mine_cells.pop(0)
            self.right_click(x, y)
            return self.make_move()  # Continue making moves
        
        # Strategy 3: Use logic to deduce safe cells
        deduced_move = self.deduce_move()
        if deduced_move:
            x, y = deduced_move
            return self.left_click(x, y)
        
        # Strategy 4: Random guess among unrevealed cells
        return self.random_guess()
    
    def update_knowledge(self):
        """Update AI's knowledge based on revealed tiles"""
        for y in range(COLS):
            for x in range(ROWS):
                tile = self.game.board.board_list[y][x]
                
                if tile.revealed and tile.type == "C":
                    # This is a number tile - we can use it for deduction
                    self.analyze_number_tile(x, y)
    
    def analyze_number_tile(self, x, y):
        """Use number tiles to deduce mines and safe cells"""
        tile = self.game.board.board_list[y][x]
        
        # Get the number from the image (simplified - you'd need to extract actual number)
        # For now, we'll use a simpler approach: check neighbors
        
        neighbors = self.get_unrevealed_neighbors(x, y)
        flagged_neighbors = self.get_flagged_neighbors(x, y)
        
        # If number of flagged neighbors equals the clue number,
        # all other neighbors are safe
        clue_number = self.get_clue_number(x, y)
        
        if len(flagged_neighbors) == clue_number:
            # All other unrevealed neighbors are safe
            for nx, ny in neighbors:
                if (nx, ny) not in flagged_neighbors:
                    if (nx, ny) not in self.safe_cells:
                        self.safe_cells.append((nx, ny))
        
        # If number of unrevealed neighbors equals clue number minus flagged,
        # then all unrevealed neighbors are mines
        if len(neighbors) == clue_number - len(flagged_neighbors):
            for nx, ny in neighbors:
                if (nx, ny) not in self.mine_cells:
                    self.mine_cells.append((nx, ny))
    
    def get_unrevealed_neighbors(self, x, y):
        """Get list of neighboring cells that aren't revealed"""
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < ROWS and 0 <= ny < COLS:
                    tile = self.game.board.board_list[ny][nx]
                    if not tile.revealed and not tile.flagged:
                        neighbors.append((nx, ny))
        return neighbors
    
    def get_flagged_neighbors(self, x, y):
        """Get list of neighboring cells with flags"""
        flagged = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < ROWS and 0 <= ny < COLS:
                    tile = self.game.board.board_list[ny][nx]
                    if tile.flagged:
                        flagged.append((nx, ny))
        return flagged
    
    def get_clue_number(self, x, y):
        """Extract the number from a clue tile"""
        tile = self.game.board.board_list[y][x]
        # This is simplified - you'd need to map the image to actual number
        # For now, we'll use the check_neighbors method
        return self.game.board.check_neighbors(x, y)
    
    def deduce_move(self):
        """Use logical deduction to find a safe move"""
        # Simple deduction: find cells that must be mines or safe
        for y in range(COLS):
            for x in range(ROWS):
                tile = self.game.board.board_list[y][x]
                if tile.revealed and tile.type == "C":
                    unrevealed = self.get_unrevealed_neighbors(x, y)
                    flagged = self.get_flagged_neighbors(x, y)
                    clue = self.get_clue_number(x, y)
                    
                    # If flagged mines already satisfy the clue, remaining are safe
                    if len(flagged) == clue and unrevealed:
                        return unrevealed[0]  # Return first safe cell
        
        return None
    
    def random_guess(self):
        """Make a random guess among unrevealed cells"""
        unrevealed = []
        for y in range(COLS):
            for x in range(ROWS):
                tile = self.game.board.board_list[y][x]
                if not tile.revealed and not tile.flagged:
                    unrevealed.append((x, y))
        
        if unrevealed:
            x, y = random.choice(unrevealed)
            return self.left_click(x, y)
        return True  # No moves left - probably won
    
    def left_click(self, x, y):
        """Simulate left click on cell"""
        # Directly call the dig method
        result = self.game.board.dig(x, y)
        
        # Update game state
        if not result:  # Hit a mine
            self.game.playing = False
            # Reveal all mines
            for row in self.game.board.board_list:
                for tile in row:
                    if tile.flagged and tile.type != "M":
                        tile.flagged = False
                        tile.revealed = True
                        tile.image = tile_not_mine
                    elif not tile.flagged and tile.type == "M":
                        tile.revealed = True
            return False
        
        # Check for win
        if self.game.check_win():
            self.game.win = True
            self.game.playing = False
            return False
        
        return True
    
    def right_click(self, x, y):
        """Simulate right click to place flag"""
        tile = self.game.board.board_list[y][x]
        if not tile.revealed:
            tile.flagged = True