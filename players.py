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

    def get_material_score(self, board):
        """Calculates immediate material value on the board.
        Positive favors White, Negative favors Black."""
        # 10 points for a Queen, 1 point for a Pawn
        white_score = (len(board.pieces(chess.PAWN, chess.WHITE)) * 1.0 + 
                       len(board.pieces(chess.QUEEN, chess.WHITE)) * 10.0)
        black_score = (len(board.pieces(chess.PAWN, chess.BLACK)) * 1.0 + 
                       len(board.pieces(chess.QUEEN, chess.BLACK)) * 10.0)
        return white_score - black_score

    def choose_action(self, env, db):
        legal_moves = list(env.board.legal_moves)
        if not legal_moves:
            return None

        # 1. Exploration phase (try something random to map out the world)
        if random.random() < self.epsilon:
            chosen_move = random.choice(legal_moves)
        else:
            # 2. Exploitation phase (Smart Evaluation)
            best_moves = []
            scratch_board = env.board.copy()
            
            if self.color == chess.WHITE:
                best_val = -float('inf')
                for move in legal_moves:
                    scratch_board.push(move)
                    
                    # Read database score
                    db_score = db.get_score(scratch_board)
                    
                    # TIE BREAKER: If database doesn't know this state (0.0), 
                    # calculate the immediate material layout score instead!
                    if db_score == 0.0:
                        score = self.get_material_score(scratch_board)
                    else:
                        score = db_score * 100.0  # Give massive priority to historical terminal wins/losses
                        
                    scratch_board.pop()
                    
                    if score > best_val:
                        best_val = score
                        best_moves = [move]
                    elif score == best_val:
                        best_moves.append(move)
            else:
                # Black wants to minimize the score
                best_val = float('inf')
                for move in legal_moves:
                    scratch_board.push(move)
                    
                    db_score = db.get_score(scratch_board)
                    
                    if db_score == 0.0:
                        score = self.get_material_score(scratch_board)
                    else:
                        score = db_score * 100.0
                        
                    scratch_board.pop()
                    
                    if score < best_val:
                        best_val = score
                        best_moves = [move]
                    elif score == best_val:
                        best_moves.append(move)
                        
            # Pick randomly out of the absolute best tied choices (prevents repeating the same pawn push)
            chosen_move = random.choice(best_moves) if best_moves else random.choice(legal_moves)

        return chosen_move.from_square * 64 + chosen_move.to_square
