import logging
import chess
from environment import PawnGameEnv
from database import PositionDatabase
from players import HumanPlayer, DBSmartPlayer
from gui import PawnGameGUI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MainApplication")

if __name__ == "__main__":
    # 1. Initialize core system modules
    db = PositionDatabase()
    env = PawnGameEnv()

    # 2. Configure players using clean OOP injection
    # Exploration_rate = 0.1 allows the bot to mix up its choices occasionally to learn new lines
    white_p = DBSmartPlayer(chess.WHITE, "DB-Smart-Bot", exploration_rate=0.1)
    black_p = HumanPlayer(chess.BLACK, "Human")

    # 3. Boot up the graphical interface layout
    logger.info("Instantiating Graphical Chess Frame Application...")
    PawnGameGUI(env, db, white_p, black_p)
    