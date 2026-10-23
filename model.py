# transformer, encoder-only
# one head!

import math
import torch

import torch.nn as nn

# pytorch layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__() # set up pytorch machinery
        position = torch.arange(max_len).float().unsqueeze(1)
        # arange() : [0,...,max_len-1] the position numbers
        # unsqueeze(1) : [[0], [1], .. [max_len-1]] of shape [max_len, 1] 
                    # to be multiplied against freqs at once in the next steps
        freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * freq) # even cols
        pe[:, 1::2] = torch.cos(position * freq) # odd cols
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]

class OneHeadAttention(nn.Module):
    # each time step looks at all the other time steps and checks whats important
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        # Comparing one step’s query with another step’s key decides how much attention to pay. 
        # The value decides what gets passed along. They start random, and training teaches them 
        # which time steps should look at which.
        self.q = nn.Linear(d_model, d_model) #query, what time steps is looking for
        self.k = nn.Linear(d_model, d_model) #key, what this timestep offers
        self.v = nn.Linear(d_model, d_model) #value, the actual value of the timestep, if paid attention
        
        self.out = nn.Linear(d_model, d_model) # recombines the mixed result 
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x): # x: [B, n, d_model]
        scores = self.q(x) @ self.k(x).transpose(1, 2) * self.scale # [B, n, n]
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(weights @ self.v(x)) # [B, n, d_model]

class EncoderLayer(nn.Module):
    # one transformer block att + fnn, each w residual and BatchNorm
    def __init__(self, d_model):
        super().__init__()
        self.attn = OneHeadAttention(d_model)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.BatchNorm1d(d_model)

    def bn(self, norm, x): 
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        x = self.bn(self.norm1, x + self.attn(x)) 
        x = self.bn(self.norm2, x + self.ffn(x)) 
        return x

class TinyTransformer(nn.Module):
    def __init__(self, num_in, num_out, d_model=16, window=24):
        super().__init__()
        self.input_linear = nn.Linear(num_in, d_model)
        self.pos = PositionalEncoding(d_model, window)
        self.encoder = EncoderLayer(d_model)
        self.output_linear = nn.Linear(d_model, num_out)

    def forward(self, x):                      # x: [B, 24, 8]
        x = self.input_linear(x)               # [B, 24, d_model]
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)                      # global average pooling over time -> [B, d_model]
        return self.output_linear(x)           # [B, num_out]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = TinyTransformer(num_in=8, num_out=1, d_model=16)
    print("output shape:", m(torch.randn(4, 24, 8)).shape)
    print("SKAB d_model=16:", count_params(m))
    print("SKAB d_model=24:", count_params(TinyTransformer(8, 1, d_model=24)))
