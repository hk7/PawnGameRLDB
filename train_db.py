import logging
import chess
import random
from tqdm import tqdm
from environment import PawnGameEnv
from database import PositionDatabase
from players import DBSmartPlayer

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")

def run_self_play_training(num_games=20000, alpha=0.15):
    db = PositionDatabase()
    env = PawnGameEnv()

    # Use a solid exploration rate during training so they thoroughly map alternatives
    white_agent = DBSmartPlayer(chess.WHITE, "Smart-White", exploration_rate=0.25)
    black_agent = DBSmartPlayer(chess.BLACK, "Smart-Black", exploration_rate=0.25)

    print(f"Starting background CORRECTED SELF-PLAY training loop for {num_games} games...")
    
    for _ in tqdm(range(num_games)):
        env.reset()
        game_history = []  # Will store pairs of (board_snapshot, whose_turn_it_was)
        game_over = False
        final_reward = 0.0

        while not game_over:
            current_turn = env.board.turn
            current_player = white_agent if current_turn == chess.WHITE else black_agent
            
            # Record state AND whose turn it was to move *from* this position
            game_history.append((chess.Board(env.board.fen()), current_turn))
            
            action = current_player.choose_action(env, db)
            if action is None:
                break
                
            _, final_reward, game_over, _, _ = env.step(action)

        # --- FIXED DECAYED BACKPROPAGATION ---
        # We value the entire game state string universally from White's perspective:
        # White wins = Positive gradient decaying backwards
        # Black wins = Negative gradient decaying backwards
        decay_factor = 0.90 
        running_reward = final_reward

        # Loop through the game history in reverse order (from last move to first move)
        for board_state, turn_color in reversed(game_history):
            # Universal alignment: update score maps clean values directly
            db.update_score(board_state, running_reward, alpha=alpha)
            
            # Decay the reward slightly for the previous step
            running_reward *= decay_factor

    db.save()
    print(f"Training complete! Database now contains {len(db.db)} unique position evaluations.")

if __name__ == "__main__":
    run_self_play_training(num_games=20000)
