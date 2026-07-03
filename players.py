import chess
import random

class Player:
    def __init__(self, color, name):
        self.color = color
        self.name = name

    def choose_move(self, env, db=None):
        raise NotImplementedError("Subclasses must implement choose_move!")


class RandomPlayer(Player):
    def choose_move(self, env, db=None):
        legal_moves = list(env.board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None


class HumanPlayer(Player):
    def choose_move(self, env, db=None):
        legal_moves = list(env.board.legal_moves)
        print(f"Legal Moves: {[m.uci() for m in legal_moves]}")
        while True:
            try:
                uci_str = input(f"[{self.name} - Enter Move (e.g. e2e4)]: ").strip()
                move = chess.Move.from_uci(uci_str)
                if move in legal_moves:
                    return move
                print("Illegal move! Try again.")
            except Exception:
                print("Invalid text format. Enter a valid move like 'f2f4'.")


class DBSmartPlayer(Player):
    def __init__(self, color, name, exploration_rate=0.1):
        super().__init__(color, name)
        self.epsilon = exploration_rate  # Chance to play a random move to explore new paths

    def choose_move(self, env, db):
        legal_moves = list(env.board.legal_moves)
        if not legal_moves:
            return None

        # 1. Exploration step (Occasional random move to discover new database states)
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        # 2. Exploitation step (Look up positions in the DB and choose the best one)
        best_move = None
        
        if self.color == chess.WHITE:
            # White wants to maximize the score
            best_val = -float('inf')
            for move in legal_moves:
                env.board.push(move)
                score = db.get_score(env.board)
                env.board.pop() # Undo simulation
                
                if score > best_val:
                    best_val = score
                    best_move = move
        else:
            # Black wants to minimize the score (make it highly negative)
            best_val = float('inf')
            for move in legal_moves:
                env.board.push(move)
                score = db.get_score(env.board)
                env.board.pop()
                
                if score < best_val:
                    best_val = score
                    best_move = move

        # Fallback security check
        return best_move if best_move else random.choice(legal_moves)
    