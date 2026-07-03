import chess
import logging

logger = logging.getLogger("Environment")

class PawnGameEnv:
    def __init__(self):
        self.board = chess.Board()
        self.reset()

    def reset(self):
        self.board.clear()
        # Set up standard 8x8 pawn ranks
        for file in range(8):
            self.board.set_piece_at(chess.square(file, 1), chess.Piece(chess.PAWN, chess.WHITE))
            self.board.set_piece_at(chess.square(file, 6), chess.Piece(chess.PAWN, chess.BLACK))
        self.board.turn = chess.WHITE
        return self.board

    def step(self, move):
        """Executes a move. Returns (board, reward, is_game_over)."""
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move attempted: {move}")

        current_turn = self.board.turn
        moving_piece = self.board.piece_at(move.from_square)
        
        # Check for immediate victory via promotion step
        is_promotion = (
            moving_piece and 
            moving_piece.piece_type == chess.PAWN and 
            chess.square_rank(move.to_square) in [0, 7]
        )

        self.board.push(move)

        # Evaluate outcomes
        if is_promotion:
            reward = 10.0 if current_turn == chess.WHITE else -10.0
            return self.board, reward, True

        # Check for immobilization (no legal moves remaining)
        if not list(self.board.legal_moves):
            # The trapped player loses!
            reward = -10.0 if self.board.turn == chess.WHITE else 10.0
            return self.board, reward, True

        return self.board, 0.0, False
    