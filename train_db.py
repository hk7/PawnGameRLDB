import logging
import chess
import random
from tqdm import tqdm  # pip install tqdm (gives a beautiful training progress bar)
from environment import PawnGameEnv
from database import PositionDatabase
from players import DBSmartPlayer, Player

# Set up logging to only print major updates so it doesn't flood the terminal
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")

class SemiRandomPunisher(Player):
    """An opponent that plays randomly UNLESS there is an immediate capture 
    or an immediate winning promotion available."""
    def __init__(self, color, name="Punisher-Bot"):
        super().__init__(color, name)
        self.is_ai = True

    def choose_action(self, env, db=None):
        legal_moves = list(env.board.legal_moves)
        if not legal_moves:
            return None

        # Priority 1: Look for an immediate winning pawn promotion
        for move in legal_moves:
            moving_piece = env.board.piece_at(move.from_square)
            if moving_piece and moving_piece.piece_type == chess.PAWN:
                if chess.square_rank(move.to_square) in [0, 7]:
                    return move.from_square * 64 + move.to_square

        # Priority 2: Look for an immediate piece capture (punish blunders!)
        captures = [m for m in legal_moves if env.board.is_capture(m)]
        if captures:
            chosen_move = random.choice(captures)
        else:
            # Priority 3: Fallback to a completely random legal move
            chosen_move = random.choice(legal_moves)

        return chosen_move.from_square * 64 + chosen_move.to_square


def run_training(num_games=10000, alpha=0.15):
    db = PositionDatabase()
    env = PawnGameEnv()

    # The agent trains as White. It has a 20% exploration rate (epsilon) 
    # during training so it discovers new positional choices.
    agent = DBSmartPlayer(chess.WHITE, "Training-Agent", exploration_rate=0.2)
    opponent = SemiRandomPunisher(chess.BLACK, "Trainer-Punisher")

    print(f"Starting background training loop for {num_games} games...")
    
    # Progress bar tracker
    for _ in tqdm(range(num_games)):
        env.reset()
        game_history = []
        game_over = False
        final_reward = 0.0

        while not game_over:
            current_player = agent if env.board.turn == chess.WHITE else opponent
            
            # Record current layout to state history before executing move
            game_history.append(chess.Board(env.board.fen()))
            
            # Fetch move action index
            action = current_player.choose_action(env, db)
            if action is None:
                break
                
            _, final_reward, game_over, _, _ = env.step(action)

        # BACKPROPAGATION: Game finished, update the database rewards along the match path
        for board_state in game_history:
            db.update_score(board_state, final_reward, alpha=alpha)

    # Save the populated database to disk
    db.save()
    print(f"Training complete! Database now contains {len(db.db)} unique position evaluations.")

if __name__ == "__main__":
    # Start with 10,000 games to build an initial baseline map
    run_training(num_games=10000)
    