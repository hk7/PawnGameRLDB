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
        """Converts the board layout into a unique text string key."""
        parts = board.fen().split(" ")
        return f"{parts[0]} {parts[1]}"

    def get_score(self, board):
        """Looks up a board position. Defaults to 0.0 if never encountered before."""
        key = self.get_position_key(board)
        return self.db.get(key, 0.0)

    def update_score(self, board, reward, alpha=0.15):
        """Applies a running average update to the position score."""
        key = self.get_position_key(board)
        current_score = self.db.get(key, 0.0)
        new_score = current_score + alpha * (reward - current_score)
        self.db[key] = round(new_score, 4)

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    self.db = json.load(f)
                logger.info(f"Loaded database with {len(self.db)} unique positions.")
            except Exception as e:
                logger.error(f"Failed to read database: {e}. Starting fresh.")
                self.db = {}
        else:
            self.db = {}

    def save(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.db, f, indent=4)
            logger.info(f"Database saved successfully. Size: {len(self.db)}")
        except Exception as e:
            logger.error(f"Failed to save database file: {e}")
            