"""A small GPT-style transformer that predicts the next chess move."""

import torch
import torch.nn.functional as F
from torch import nn

from chess.tokenizer import VOCAB_SIZE

HEAD_SIZE = 64  # size of each attention head; the model width must be a multiple of it

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

        att = (Q@K.transpose(-1, -2))/self.dk**0.5  #(B, h, T, dk)@(B, h, dk, T) = (B, h, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = att.softmax(-1)@V                     #(B, h, T, T)@(B, h, T, dk)  = (B, h, T, dk)

        att = att.transpose(1,2).reshape(B, T, -1)  #(B, T, d)

        return self.o(att) #(B, T, d)

class FeedForward(nn.Module):
    #un réseau Feed Forward
    def __init__(self, d, mult = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, mult*d), nn.GELU(), nn.Linear(mult*d, d)
        )

    def forward(self, x):
        return self.net(x)

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
    def __init__(self, vocab_size, n_blocks, d, n_heads, max_len): #max_len >= SEQ_LEN = 256
        super().__init__() 
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(n_heads = n_heads, d = d) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos) # (B, T, d)
        mask = torch.tril(torch.ones(T, T, device=tokens.device))
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)

## The transformer follows the GPT 2 architecture

