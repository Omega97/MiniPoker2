from pathlib import Path


# __file__ is the location of this specific script.
# .resolve() makes it absolute. .parent is the folder.
# We call .parent twice to go from mini_poker/paths.py -> mini_poker/ -> ProjectRoot/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Define subdirectories
DATA_DIR = PROJECT_ROOT / "instances"
GAME_DATA_DIR = PROJECT_ROOT / "game_data"


# Ensure directories exist so the agent doesn't crash on save
for folder in (DATA_DIR, GAME_DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)
