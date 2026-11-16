"""Loading the smolchess dataset, tokenizing it once, and serving batches."""

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from chess.tokenizer import PAD_ID, encode
DATASET_NAME = "dlouapre/smolchess"
SEQ_LEN = 256  # max tokens per game, <bos> and <eos> included; longer games are cut
CACHE_DIR = Path("data_cache")  # folder where the tokenized games are saved


def load_tokens(config: str, split: str) -> np.ndarray:
    """Return the tokenized games of a split, as an array of shape (num_games, SEQ_LEN).

    config: "1M" (1 million games) or "default" (all 66M games)
    split: "train", "validation" or "strong"

    The first call downloads and tokenizes the games, then saves the result.
    The next calls simply read the saved file.
    """
    path = CACHE_DIR / f"{config}_{split}_{SEQ_LEN}.npy"
    if path.exists():
        return np.load(path)

    games = load_dataset(DATASET_NAME, config, split=split)

    # One row per game, filled with <pad> to start with.
    # uint16 stores integers up to 65,535: enough for 1,839 tokens,
    # and 4x lighter in memory than the default int64.
    tokens = np.full((len(games), SEQ_LEN), PAD_ID, dtype=np.uint16)

    for i, moves in enumerate(tqdm(games["moves"], desc=f"Tokenizing {split}")):
        ids = encode(moves)[:SEQ_LEN]  # cut the game if it is too long
        tokens[i, : len(ids)] = ids  # write it at the start of the row; the rest stays <pad>

    CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, tokens)
    return tokens


def iterate_batches(tokens: np.ndarray, batch_size: int, device: torch.device, shuffle: bool = True):
    """Go once through all the games, yielding batches of shape (batch_size, length)."""
    # Shuffle the order of the games so that each batch mixes different games
    if shuffle:
        order = np.random.permutation(len(tokens))
    else:
        order = np.arange(len(tokens))

    for start in range(0, len(order) - batch_size + 1, batch_size):
        rows = order[start : start + batch_size]
        # PyTorch needs int64 to look up the embedding table
        batch = torch.from_numpy(tokens[rows].astype(np.int64))

        # If the longest game in this batch is 90 tokens long, the columns 90 to 255
        # contain only <pad>: drop them, the model would compute them for nothing.
        length = int((batch != PAD_ID).sum(dim=1).max())
        yield batch[:, :length].to(device)

