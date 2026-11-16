"""A small GPT-style transformer that predicts the next chess move."""

import torch
import torch.nn.functional as F
from torch import nn

from chess.tokenizer import VOCAB_SIZE

ROPE_BASE = 10_000 #standard value from the paper 

def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding: rotate each query or key according to its position.

    x: (batch, heads, length, head_dim) -> same shape. head_dim must be even.
    """
    length, size = x.shape[-2], x.shape[-1]
    half = size // 2

    # One rotation speed per pair of dimensions: fast for the first pairs, slow for the last ones
    frequencies = ROPE_BASE ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    positions = torch.arange(length, device=x.device, dtype=torch.float32)  # 0, 1, 2, ...
    angles = positions[:, None] * frequencies[None, :]  # (length, half): angle = position * speed
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    # Split each vector in two halves and rotate each pair (x1, x2) by its angle,
    # like rotating a point in 2D: (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, n_heads, d):
        super().__init__()
        self.h = n_heads
        self.dk = d // n_heads
        self.q, self.k, self.v = nn.Linear(d, d, bias = False), nn.Linear(d, d, bias = False), nn.Linear(d, d, bias = False)
        self.o = nn.Linear(d, d)

    def forward(self, x, mask):
        B, T, d = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2) #(B, h, T, dk)
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2) #(B, h, T, dk)
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2) #(B, h, T, dk)

        Q, K = rope(Q), rope(K)

        att = (Q@K.transpose(-1, -2))/self.dk**0.5  #(B, h, T, dk)@(B, h, dk, T) = (B, h, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = att.softmax(-1)@V                     #(B, h, T, T)@(B, h, T, dk)  = (B, h, T, dk)

        att = att.transpose(1,2).reshape(B, T, -1)  #(B, T, d)

        return self.o(att) #(B, T, d)

class FeedForward(nn.Module):
    #un réseau Feed Forward
    def __init__(self, d):
        super().__init__()
        hidden = 8 * d // 3
        self.up = nn.Linear(d, hidden, bias  = False)
        self.gate = nn.Linear(d, hidden, bias = False)
        self.down = nn.Linear(hidden, d, bias = False)


    def forward(self, x):
        return self.down(F.silu(self.gate(x))*self.up(x))

class Block(nn.Module):
    #Un block de transformer = FeedForward + Attention
    def __init__(self, n_heads, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = Attention(n_heads, d)
        self.feedforward = FeedForward(d)

    def forward(self, x, mask = None):
        x = x + self.attention(self.ln1(x), mask) #attention with preNorm + skip connection
        x = x + self.feedforward(self.ln2(x)) # feedforward with prenorm + skip connection

        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, n_blocks, d, n_heads): #max_len >= SEQ_LEN = 256
        super().__init__() 
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([Block(n_heads = n_heads, d = d) for _ in range(n_blocks)])
        self.rmsn = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.tok_emb(tokens)  # (B, T, d): positions are handled by RoPE in the attention
        mask = torch.tril(torch.ones(T, T, device=tokens.device))
        for block in self.blocks:
            x = block(x, mask)

        x = self.rms(x)
        return self.head(x)

## The transformer follows the GPT 2 architecture

