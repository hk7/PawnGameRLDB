import tkinter as tk
from tkinter import messagebox
import chess
import logging
import traceback
from environment import PawnGameEnv
from database import PositionDatabase

logger = logging.getLogger("GameGUI")

class PawnGameGUI:
    def __init__(self, env: PawnGameEnv, db: PositionDatabase, white_player, black_player):
        self.env = env
        self.db = db
        self.players = {chess.WHITE: white_player, chess.BLACK: black_player}
        
        self.history = []  # Stores (board, turn)
        self.game_over = False

        self.root = tk.Tk()
        self.root.title("Pawn Game - Reinforcement Learning DB")

        self.square_size = 70
        self.canvas = tk.Canvas(self.root, width=8 * self.square_size, height=8 * self.square_size)
        self.canvas.pack()

        # Re-added clean high-visibility unicode definitions
        self.piece_symbols = {
            'P': '♙',  # White Pawn
            'p': '♟',  # Black Pawn
            'Q': '♕',  # White Queen
            'q': '♛',  # Black Queen
        }

        self.selected_square = None
        self.canvas.bind("<Button-1>", self.on_square_click)

        self.draw_board()
        self.root.after(500, self.check_ai_turn)
        self.root.mainloop()

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#f0d9b5", "#b58863"]

        for row in range(8):
            for col in range(8):
                x1, y1 = col * self.square_size, row * self.square_size
                x2, y2 = x1 + self.square_size, y1 + self.square_size
                chess_square = chess.square(col, 7 - row)

                if self.selected_square == chess_square:
                    color = "#7b9c60"
                else:
                    color = colors[(row + col) % 2]

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                piece = self.env.board.piece_at(chess_square)
                if piece:
                    # symbol = '♙' if piece.color == chess.WHITE else '♟'
                    # if piece.piece_type == chess.QUEEN:
                    #     symbol = '♕' if piece.color == chess.WHITE else '♛'

                    # Correctly grab the exact symbol lookup character (P, p, Q, or q)
                    symbol = self.piece_symbols.get(piece.symbol(), piece.symbol())

                    text_color = "#000000" if piece.color == chess.BLACK else "#ffffff"
                    self.canvas.create_text(
                        x1 + self.square_size/2, y1 + self.square_size/2,
                        text=symbol, font=("Courier", 36, "bold"), fill=text_color
                    )

    def on_square_click(self, event):
        if self.game_over:
            return

        current_turn = self.env.board.turn
        if getattr(self.players[current_turn], "is_ai", False):
            return

        col = event.x // self.square_size
        row = event.y // self.square_size
        clicked_square = chess.square(col, 7 - row)

        if self.selected_square is None:
            piece = self.env.board.piece_at(clicked_square)
            if piece and piece.color == current_turn:
                self.selected_square = clicked_square
                self.draw_board()
        else:
            move = chess.Move(self.selected_square, clicked_square)
            if self.env.board.piece_at(self.selected_square) and self.env.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) in [0, 7]:
                    move.promotion = chess.QUEEN

            legal_moves = list(self.env.board.legal_moves)
            if move in legal_moves:
                action_idx = move.from_square * 64 + move.to_square
                
                # Save turn details
                self.history.append((chess.Board(self.env.board.fen()), current_turn))

                try:
                    _, reward, terminated, _, _ = self.env.step(action_idx)
                    self.selected_square = None
                    self.draw_board()

                    if terminated:
                        self.game_over = True
                        self.end_game(reward)
                        return

                    self.root.after(300, self.check_ai_turn)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    traceback.print_exc()
            else:
                self.selected_square = None
                self.draw_board()

    def check_ai_turn(self):
        if self.game_over or self.env.board.is_game_over() or not list(self.env.board.legal_moves):
            return

        current_turn = self.env.board.turn
        player = self.players[current_turn]

        if getattr(player, "is_ai", False):
            try:
                self.history.append((chess.Board(self.env.board.fen()), current_turn))
                action = player.choose_action(self.env, self.db)
                
                if action is None:
                    return

                _, reward, terminated, _, _ = self.env.step(action)
                self.draw_board()

                if terminated:
                    self.game_over = True
                    self.end_game(reward)
                    return

                self.root.after(300, self.check_ai_turn)
            except Exception as e:
                logger.error(f"AI Failure: {e}")
                traceback.print_exc()

    def end_game(self, final_reward):
        logger.info(f"Processing turn-aware backpropagation across {len(self.history)} states.")
        for board_state, turn_color in self.history:
            if turn_color == chess.WHITE:
                self.db.update_score(board_state, final_reward)
            else:
                self.db.update_score(board_state, -final_reward)
        self.db.save()

        if final_reward > 0:
            outcome = "White Wins!"
        elif final_reward < 0:
            outcome = "Black Wins!"
        else:
            outcome = "Draw Game!"

        popup = tk.Toplevel(self.root)
        popup.title("Game Over")
        tk.Label(popup, text=outcome, font=("Helvetica", 16, "bold"), padx=20, pady=20).pack()
        tk.Button(popup, text="Close", command=self.root.quit).pack(pady=10)
