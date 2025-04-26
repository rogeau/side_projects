import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import configs

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=configs.EMBED_DIM, max_length=configs.MAX_LENGTH):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)

    def forward(self, x):
        bs, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_embeddings = self.pos_embed(positions)
        token_embeddings = self.token_embed(x)
        return token_embeddings + pos_embeddings 

class MLP(nn.Module):
    def __init__(self, embed_dim, forward_expansion=4, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*forward_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*forward_expansion, embed_dim),
            nn.Dropout(dropout)
        )

        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return x + self.mlp(self.layernorm(x))


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=16, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (embed_dim // heads) ** -0.5

        self.norm = nn.LayerNorm(embed_dim)

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        residual = x

        x = self.norm(x)

        context = x if context is None else context

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b nq (h d) -> b h nq d', h=self.heads)
        k = rearrange(k, 'b nk (h d) -> b h nk d', h=self.heads)
        v = rearrange(v, 'b nv (h d) -> b h nv d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if context is x:
            q_len, k_len = q.size(2), k.size(2)
            mask = torch.tril(torch.ones(q_len, k_len, dtype=torch.bool, device=x.device))
            dots = dots.masked_fill(~mask, float('-inf'))

        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return residual + self.to_out(out)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=configs.EMBED_DIM, depth=configs.BLOCK_NUMBER, dropout=0.):
        super().__init__()
        self.vocab_size = vocab_size

        self.embed = Embedding(self.vocab_size)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(embed_dim),
                Attention(embed_dim),
                MLP(embed_dim)
            ]))


        self.attention = Attention(embed_dim)
        self.mlp = MLP(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, caption, feature):
        x = self.embed(caption)

        for self_attention, cross_attention, mlp in self.layers:
            x = self_attention(x)
            x = cross_attention(x, feature)
            x = mlp(x)


        logits = self.fc(x)
        probs = self.softmax(logits)

        return logits, probs

if __name__ == "__main__":
    image = torch.randn(4, 196, 1024)

    caption = torch.randint(0, 8000, (4, 50))
    decoder = Decoder(vocab_size = 8000)
    x, _ = decoder(caption, image)
    print(x.shape)