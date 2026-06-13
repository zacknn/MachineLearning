# Minesweeper with AI

An interactive Minesweeper game built with Python and Pygame, featuring both human and AI player modes.

## Features

- **Interactive GUI**: Play classic Minesweeper with a graphical interface
- **Human Mode**: Click tiles to reveal them, right-click to flag suspected mines
- **AI Mode**: Watch an intelligent AI player solve the puzzle automatically
- **Smart AI Strategy**: The AI uses multiple strategies to make optimal moves:
  - Deduction logic based on numbered tiles
  - Pattern recognition for mine detection
  - Random guessing as fallback
- **Customizable Game Settings**: Adjust board size, mine count, and tile appearance

## Game Rules

- **Left Click**: Reveal a tile
- **Right Click**: Flag/unflag a tile as a mine
- **Goal**: Reveal all non-mine tiles without hitting a mine
- **Numbered Tiles**: Show how many mines are adjacent to that tile

## File Structure

```
Minesweeper/
├── main.py              # Entry point of the game
├── ui_main.py           # Main game loop and UI logic
├── settings.py          # Game configuration and constants
├── sprites.py           # Tile and board classes
├── button.py            # Button UI component
├── ai_player.py         # AI player logic and strategies
├── assets/              # Game sprites (tile images)
└── README.md            # This file
```

## Installation

### Requirements

- Python 3.7+
- Pygame 2.x

### Setup

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install pygame
   ```

## Usage

Run the game:

```bash
python3 main.py
```

Or directly:

```bash
python3 ui_main.py
```

### Game Controls

- **PLAY Button**: Start a new game in human mode
- **AI PLAY Button**: Start a new game and let the AI play automatically
- **Left Click**: Reveal a tile (human mode)
- **Right Click**: Flag/unflag a tile (human mode)
- **Close Window**: Exit the game

## Game Modes

### Human Mode
- Click the **PLAY** button to start
- Reveal tiles by left-clicking
- Flag suspected mines with right-click
- Win by revealing all non-mine tiles

### AI Mode
- Click the **AI PLAY** button to start
- Watch the AI automatically solve the puzzle
- The AI makes a move every 500ms (configurable)
- Observe its deduction strategy in action

## Configuration

Edit `settings.py` to customize:

- **TILE_SIZE**: Size of each tile in pixels (default: 32)
- **ROWS / COLS**: Board dimensions (default: 15x15)
- **AMOUNT_MINES**: Number of mines to place (default: 20)
- **FPS**: Game frame rate (default: 60)
- **Colors**: Customize background and button colors

## AI Strategy

The AI player uses a sophisticated decision-making process:

1. **Update Knowledge**: Analyzes all revealed numbered tiles
2. **Deduction**: Uses logic to identify guaranteed safe or mine cells
3. **Pattern Recognition**: Detects mine clusters and safe zones
4. **Random Guess**: Falls back to random selection when logic is inconclusive

## Project Structure

### Core Classes

- **Game**: Main game controller handling events, updates, and rendering
- **Board**: 2D grid of tiles with game logic
- **Tile**: Individual tile with state (revealed, flagged, type)
- **Button**: Clickable UI buttons for mode selection
- **AIPlayer**: Intelligent bot with deduction algorithms


### Missing Assets
Check that the `assets/` folder contains all required tile images:
- TileEmpty.png, Tile1.png through Tile8.png
- TileExploded.png, TileFlag.png, TileMine.png, etc.

### Performance Issues
- Reduce FPS if experiencing lag
- Decrease board size or AI move speed
- Update Pygame and ensure hardware acceleration is enabled

## Future Enhancements

- Difficulty levels (easy, medium, hard)
- Score/time tracking
- Undo functionality
- Different board shapes
- Improved AI with machine learning

## License

This project is part of the MachineLearning portfolio.

## Author

Created as an AI player project combining game development with algorithm optimization.
