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
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                color = "#DDBB88" if (rank + file) % 2 == 0 else "#996633"
                
                # Highlight selected square
                if self.selected_square == chess.square(file, rank):
                    color = "#FFFF99"
                    
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                piece = self.env.board.piece_at(chess.square(file, rank))
                if piece:
                    symbol = self.piece_symbols.get(piece.symbol(), "")
                    self.canvas.create_text(x1 + self.square_size/2, y1 + self.square_size/2,
                                           text=symbol, font=("Arial", 32), fill="black")

    def on_square_click(self, event):
        if self.game_over:
            return
            
        current_turn = self.env.board.turn
        if self.players[current_turn].is_ai:
            return  # Ignore clicks when it's the AI's turn

        file = event.x // self.square_size
        rank = 7 - (event.y // self.square_size)
        clicked_square = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.env.board.piece_at(clicked_square)
            if piece and piece.color == current_turn:
                self.selected_square = clicked_square
                self.draw_board()
        else:
            # Attempting a move from selected_square to clicked_square
            from_sq = self.selected_square
            to_sq = clicked_square
            action_idx = from_sq * 64 + to_sq
            
            # Verify if this matches an actual legal move
            matched_move = None
            for move in self.env.board.legal_moves:
                if move.from_square == from_sq and move.to_square == to_sq:
                    matched_move = move
                    break
            
            if matched_move:
                from_str = chess.square_name(from_sq)
                to_str = chess.square_name(to_sq)
                color_str = "White" if current_turn == chess.WHITE else "Black"
                
                # --- ADDED LOGGING FOR HUMAN PLAYER MOVE ---
                print(f"[POST-MORTEM LOG] {color_str} (Human) played: {from_str} -> {to_str} (UCI: {matched_move.uci()})")
                logger.info(f"Human Move: {matched_move.uci()}")
                
                # Save snapshot to training history before step execution
                self.history.append((chess.Board(self.env.board.fen()), current_turn))
                
                # Execute move transaction
                _, reward, self.game_over, _, _ = self.env.step(action_idx)
                
                self.selected_square = None
                self.draw_board()
                
                if self.game_over:
                    self.end_game(reward)
                    return
                    
                self.root.after(300, self.check_ai_turn)
            else:
                # Reset selection if invalid move square clicked
                self.selected_square = None
                self.draw_board()

    def check_ai_turn(self):
        if self.game_over:
            return
            
        current_turn = self.env.board.turn
        player = self.players[current_turn]
        
        if player.is_ai:
            try:
                self.history.append((chess.Board(self.env.board.fen()), current_turn))
                
                action = player.choose_action(self.env, self.db)
                if action is None:
                    return
                
                from_sq = action // 64
                to_sq = action % 64
                from_str = chess.square_name(from_sq)
                to_str = chess.square_name(to_sq)
                color_str = "White" if current_turn == chess.WHITE else "Black"
                
                # --- ADDED LOGGING FOR AI PLAYER MOVE ---
                print(f"[POST-MORTEM LOG] {color_str} (AI) played: {from_str} -> {to_str}")
                
                _, reward, self.game_over, _, _ = self.env.step(action)
                self.draw_board()
                
                if self.game_over:
                    self.end_game(reward)
                    return
                    
                self.root.after(300, self.check_ai_turn)
            except Exception as e:
                logger.error(f"AI Failure: {e}")
                traceback.print_exc()

    def end_game(self, final_reward):
        logger.info(f"Processing turn-aware decayed backpropagation across {len(self.history)} states.")
        decay_factor = 0.90
        running_reward = final_reward

        for board_state, turn_color in reversed(self.history):
            self.db.update_score(board_state, running_reward)
            running_reward *= decay_factor
            
        self.db.save()

        if final_reward > 0:
            outcome = "White Wins!"
        elif final_reward < 0:
            outcome = "Black Wins!"
        else:
            outcome = "Draw Game!"

        popup = tk.Toplevel(self.root)
        popup.title("Game Over")
        tk.Label(popup, text=outcome, font=("Arial", 16, "bold"), padx=20, pady=20).pack()
        tk.Button(popup, text="OK", command=self.root.destroy, width=10).pack(pady=10)
