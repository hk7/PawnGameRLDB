import json
import os
import logging

logger = logging.getLogger("Database")

class PositionDatabase:
    def __init__(self, filename="pawn_game_db.json"):
        self.filename = filename
        self.db = {}
        self.load()

    def get_position_key(self, board):
        """Converts the board layout into a unique string key (using standard FEN)."""
        # We split to only grab the piece layout and whose turn it is
        parts = board.fen().split(" ")
        return f"{parts[0]} {parts[1]}"

    def get_score(self, board):
        """Returns the score of a position. Defaults to 0.0 if never seen before."""
        key = self.get_position_key(board)
        return self.db.get(key, 0.0)

    def update_score(self, board, reward, alpha=0.1):
        """
        Updates the score using a running average (Q-learning rule).
        alpha is the learning rate (how fast it updates its mind).
        """
        key = self.get_position_key(board)
        current_score = self.db.get(key, 0.0)
        # Running average formula
        new_score = current_score + alpha * (reward - current_score)
        self.db[key] = round(new_score, 4)

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    self.db = json.load(f)
                logger.info(f"Loaded database with {len(self.db)} known positions.")
            except Exception as e:
                logger.error(f"Failed to load database: {e}. Starting fresh.")
                self.db = {}
        else:
            self.db = {}

    def save(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.db, f, indent=4)
            logger.info(f"Database successfully saved. Total positions: {len(self.db)}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            