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
    def __init__(self, color, name="DB Bot", exploration_rate=0.0):
        super().__init__(color, name)
        self.is_ai = True
        self.epsilon = exploration_rate

    def get_material_score(self, board):
        """Fallback material counting evaluation from White's perspective."""
        white_score = (len(board.pieces(chess.PAWN, chess.WHITE)) * 1.0 + 
                       len(board.pieces(chess.QUEEN, chess.WHITE)) * 10.0)
        black_score = (len(board.pieces(chess.PAWN, chess.BLACK)) * 1.0 + 
                       len(board.pieces(chess.QUEEN, chess.BLACK)) * 10.0)
        return white_score - black_score

    def evaluate_node(self, board, db):
        """Evaluates a leaf node board state cleanly from White's perspective."""
        if board.is_game_over() or not list(board.legal_moves):
            return 1000.0 if self.get_material_score(board) > 0 else -1000.0
            
        db_score = db.get_score(board)
        if db_score != 0.0:
            return db_score * 100.0
        return self.get_material_score(board)

    def choose_action(self, env, db):
        legal_moves = list(env.board.legal_moves)
        if not legal_moves:
            return None

        # Random exploration check
        if random.random() < self.epsilon:
            random_move = random.choice(legal_moves)
            return random_move.from_square * 64 + random_move.to_square

        best_moves = []
        scratch_board = env.board.copy()

        if self.color == chess.WHITE:
            best_val = -float('inf')
            for move in legal_moves:
                scratch_board.push(move)
                
                # White checks Black's best replies (minimizing White's score)
                black_replies = list(scratch_board.legal_moves)
                if not black_replies:
                    move_val = self.evaluate_node(scratch_board, db)
                else:
                    worst_case_for_white = float('inf')
                    for b_move in black_replies:
                        scratch_board.push(b_move)
                        reply_score = self.evaluate_node(scratch_board, db)
                        scratch_board.pop()
                        if reply_score < worst_case_for_white:
                            worst_case_for_white = reply_score
                    move_val = worst_case_for_white
                    
                scratch_board.pop()

                if move_val > best_val:
                    best_val = move_val
                    best_moves = [move]
                elif move_val == best_val:
                    best_moves.append(move)
        else:
            # Black wants to minimize White's score advantages
            best_val = float('inf')
            for move in legal_moves:
                scratch_board.push(move)
                
                # Black checks White's best replies (maximizing White's score)
                white_replies = list(scratch_board.legal_moves)
                if not white_replies:
                    move_val = self.evaluate_node(scratch_board, db)
                else:
                    worst_case_for_black = -float('inf')
                    for w_move in white_replies:
                        scratch_board.push(w_move)
                        reply_score = self.evaluate_node(scratch_board, db)
                        scratch_board.pop()
                        if reply_score > worst_case_for_black:
                            worst_case_for_black = reply_score
                    move_val = worst_case_for_black
                    
                scratch_board.pop()

                if move_val < best_val:
                    best_val = move_val
                    best_moves = [move]
                elif move_val == best_val:
                    best_moves.append(move)

        chosen_move = random.choice(best_moves) if best_moves else random.choice(legal_moves)
        return chosen_move.from_square * 64 + chosen_move.to_square
