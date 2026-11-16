"""Tokenizer for chess games in UCI notation: one token per move."""

# Board coordinates 
FILES = "abcdefgh"  # columns
RANKS = "12345678"  # rows
# All 64 squares, indexed from 0 (a1) to 63 (h8): index = 8 * rank + file
SQUARES = [file + rank for rank in RANKS for file in FILES]

# Special tokens 
# <pad>: fills the end of short games so that all games in a batch have the same length
# <bos>: "beginning of sequence", marks the start of a game
# <eos>: "end of sequence", marks the end of a game
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def all_moves() -> list[str]:
    """Every move that is geometrically possible on an empty board.

    A queen move (straight line or diagonal) or a knight move covers every piece:
    king, rook, bishop and pawn moves are all queen moves, and castling is written
    as the king's move (e1g1). Queen promotions are added at the end.
    """
    moves = []
    for start in range(64):
        for end in range(64):
            # Distance between the two squares, in columns and in rows
            file_gap = abs(end % 8 - start % 8)
            rank_gap = abs(end // 8 - start // 8)
            # Same column, same row, or same diagonal
            queen_move = file_gap == 0 or rank_gap == 0 or file_gap == rank_gap
            # An "L" shape: 1 square one way, 2 squares the other way
            knight_move = {file_gap, rank_gap} == {1, 2}
            if start != end and (queen_move or knight_move):
                moves.append(SQUARES[start] + SQUARES[end])

    # Promotions: a pawn reaching the last rank, straight ahead or capturing diagonally.
    # The dataset only keeps promotions to a queen, written with a final "q".
    for start_file in range(8):
        for end_file in range(max(0, start_file - 1), min(8, start_file + 2)):
            moves.append(f"{FILES[start_file]}7{FILES[end_file]}8q")  # white pawn
            moves.append(f"{FILES[start_file]}2{FILES[end_file]}1q")  # black pawn
    return moves


#  Vocabulary 
VOCAB = SPECIAL_TOKENS + all_moves()  # id -> token
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCAB)}  # token -> id
VOCAB_SIZE = len(VOCAB)
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2  # the special tokens come first in VOCAB


def encode(moves: str) -> list[int]:
    """Turn a game into token ids: 'e2e4 e7e5' -> [BOS_ID, id(e2e4), id(e7e5), EOS_ID]"""
    return [BOS_ID] + [TOKEN_TO_ID[move] for move in moves.split()] + [EOS_ID]


def decode(ids: list[int]) -> list[str]:
    """Turn token ids back into moves (inverse of encode)."""
    return [VOCAB[i] for i in ids]