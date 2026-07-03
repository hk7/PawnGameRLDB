import chess
import random

class Player:
    def __init__(self, color, name):
        self.color = color
        self.name = name
        self.is_ai = False

    def choose_action(self, env, db=None):
        raise NotImplementedError("Subclasses must override choose_action!")


class HumanPlayer(Player):
    def __init__(self, color, name="Human"):
        super().__init__(color, name)
        self.is_ai = False

    def choose_action(self, env, db=None):
        return None


class DBSmartPlayer(Player):
    def __init__(self, color, name="DB Bot", exploration_rate=0.1):
        super().__init__(color, name)
        self.is_ai = True
        self.epsilon = exploration_rate

    def choose_action(self, env, db):
        legal_moves = list(env.board.legal_moves)
        if not legal_moves:
            return None

        # 1. Exploration phase (random variation mapping)
        if random.random() < self.epsilon:
            chosen_move = random.choice(legal_moves)
        else:
            # 2. Exploitation phase
            best_move = None
            
            # Create a completely isolated copy of the board for simulation scratchpad work
            # This completely protects en passant states from leaking or crashing the main cycle
            scratch_board = env.board.copy()
            
            if self.color == chess.WHITE:
                best_val = -float('inf')
                for move in legal_moves:
                    scratch_board.push(move)
                    score = db.get_score(scratch_board)
                    scratch_board.pop()
                    
                    if score > best_val:
                        best_val = score
                        best_move = move
            else:
                best_val = float('inf')
                for move in legal_moves:
                    scratch_board.push(move)
                    score = db.get_score(scratch_board)
                    scratch_board.pop()
                    
                    if score < best_val:
                        best_val = score
                        best_move = move
                        
            chosen_move = best_move if best_move else random.choice(legal_moves)

        # Secure action index calculation alignment
        return chosen_move.from_square * 64 + chosen_move.to_square
