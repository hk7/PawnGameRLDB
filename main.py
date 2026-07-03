import chess
import logging
from environment import PawnGameEnv
from database import PositionDatabase
from players import RandomPlayer, HumanPlayer, DBSmartPlayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def play_match():
    db = PositionDatabase()
    env = PawnGameEnv()

    # Assign players
    white_player = DBSmartPlayer(chess.WHITE, "DB-Smart-Bot", exploration_rate=0.0) # 0.0 exploration means pure smart play
    black_player = HumanPlayer(chess.BLACK, "Human")

    print("\n=== Match Started ===")
    print(env.board)
    
    game_over = False
    history = []  # Track positions seen this match to backpropagate final score

    while not game_over:
        current_player = white_player if env.board.turn == chess.WHITE else black_player
        
        # Save current state into history before moving
        history.append(chess.Board(env.board.fen()))
        
        # Choose and execute move
        move = current_player.choose_move(env, db)
        _, final_reward, game_over = env.step(move)
        
        print(f"\n{current_player.name} played {move}")
        print(env.board)

    print(f"\nMatch Ended! Final reward outcome: {final_reward}")

    # BACKPROPAGATION: Update the database for all positions seen in this game
    # If White wins, all steps along that path get an upward bump in value.
    logger.info("Updating database scores based on game outcome...")
    for board_state in history:
        db.update_score(board_state, final_reward, alpha=0.1)
        
    db.save()

if __name__ == "__main__":
    play_match()
    