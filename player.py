import chess
from ChessCNN import ChessCNN
import torch
import tkinter as tk
from tkinter import simpledialog, messagebox

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game")
        
        self.board = chess.Board()
        self.engine = ChessCNN()
        self.engine.load_state_dict(torch.load("chess_model_early_stopping.pth"))
        self.engine.eval()

        self.label = tk.Label(root, text="Welcome to the chess game\nYou are playing as black.\nEnter your moves in SAN notation.\nExample: e4, Nf3, Bb5, O-O, etc.\nType 'exit' to quit the game.")
        self.label.pack()

        self.text = tk.Text(root, height=20, width=50)
        self.text.pack()
        self.update_board()

        self.entry = tk.Entry(root)
        self.entry.pack()
        self.entry.bind("<Return>", self.on_enter)

        if self.board.turn == chess.WHITE:
            self.ai_move()

    def update_board(self):
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, str(self.board))

    def on_enter(self, event):
        move = self.entry.get()
        self.entry.delete(0, tk.END)
        if move == "exit":
            self.root.quit()
        else:
            try:
                self.board.push_san(move)
                self.update_board()
                if not self.board.is_game_over():
                    self.ai_move()
                else:
                    messagebox.showinfo("Game Over", self.board.result())
            except ValueError:
                messagebox.showerror("Invalid Move", "Invalid move. Try again.")

    def ai_move(self):
        with torch.no_grad():
            move = self.engine.predict(self.board)
        self.board.push(move)
        self.update_board()
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", self.board.result())

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()