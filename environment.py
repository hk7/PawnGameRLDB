import chess
import logging

logger = logging.getLogger("Environment")

class PawnGameEnv:
    def __init__(self):
        self.board = chess.Board()
        self.reset()

    def reset(self):
        self.board.clear()
        # Set up standard pawn rows
        for file in range(8):
            self.board.set_piece_at(chess.square(file, 1), chess.Piece(chess.PAWN, chess.WHITE))
            self.board.set_piece_at(chess.square(file, 6), chess.Piece(chess.PAWN, chess.BLACK))
        self.board.turn = chess.WHITE
        return self.board

    def step(self, action_idx):
        """Executes a move using an action index to remain uniform with your GUI hooks."""
        from_sq = action_idx // 64
        to_sq = action_idx % 64
        move = chess.Move(from_sq, to_sq)

        # Automatic queen promotion validation
        moving_piece = self.board.piece_at(from_sq)
        if moving_piece and moving_piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
            move.promotion = chess.QUEEN

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move attempted: {move}")

        current_turn = self.board.turn
        self.board.push(move)

        # 1. Check for immediate promotion victory
        if moving_piece and moving_piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
            reward = 10.0 if current_turn == chess.WHITE else -10.0
            return self.board, reward, True, False, {}

        # 2. Check for immobilization (No moves remaining means trapped player loses)
        if not list(self.board.legal_moves):
            reward = -10.0 if self.board.turn == chess.WHITE else 10.0
            return self.board, reward, True, False, {}

        # 3. Fallback standard check
        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome and outcome.winner == chess.WHITE:
                return self.board, 10.0, True, False, {}
            elif outcome and outcome.winner == chess.BLACK:
                return self.board, -10.0, True, False, {}

        return self.board, 0.0, False, False, {}
    