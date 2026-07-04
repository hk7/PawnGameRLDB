import logging
import chess
import random
from tqdm import tqdm  # pip install tqdm
from environment import PawnGameEnv
from database import PositionDatabase
from players import DBSmartPlayer

# Set up logging to only print major updates so it doesn't flood the terminal
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")

def run_self_play_training(num_games=20000, alpha=0.15):
    db = PositionDatabase()
    env = PawnGameEnv()

    # BOTH players are now Smart DB Bots! 
    # We give them a decent exploration_rate (epsilon=0.25) so they try 
    # new lines and variations, preventing them from playing the exact same game over and over.
    white_agent = DBSmartPlayer(chess.WHITE, "Smart-White", exploration_rate=0.25)
    black_agent = DBSmartPlayer(chess.BLACK, "Smart-Black", exploration_rate=0.25)

    print(f"Starting background SELF-PLAY training loop for {num_games} games...")
    
    for _ in tqdm(range(num_games)):
        env.reset()
        game_history = []
        game_over = False
        final_reward = 0.0

        while not game_over:
            current_player = white_agent if env.board.turn == chess.WHITE else black_agent
            
            # Record current layout to state history before executing move
            game_history.append(chess.Board(env.board.fen()))
            
            # Fetch move action index using the shared database
            action = current_player.choose_action(env, db)
            if action is None:
                break
                
            _, final_reward, game_over, _, _ = env.step(action)

        # BACKPROPAGATION: Rewind through the game and update valuations
        # Since it's self-play, it updates all the moves that led to this outcome
        for board_state in game_history:
            db.update_score(board_state, final_reward, alpha=alpha)

    # Save the populated database to disk
    db.save()
    print(f"Training complete! Database now contains {len(db.db)} unique position evaluations.")

if __name__ == "__main__":
    # We increase the volume to 20,000 games because self-play explores deeper, higher-quality tactical paths
    run_self_play_training(num_games=20000)
