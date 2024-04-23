import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class NeuralLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_sz,
        embedding_size,
        lin_size,
        context_size = 3,
        nonlin = nn.functional.tanh,
    ):
        super().__init__()

        self.vocab_sz = vocab_sz
        self.embedding_size = embedding_size
        self.lin_size = lin_size
        self.context_size = context_size
        self.embedding = nn.Embedding(self.vocab_sz, self.embedding_size)
        self.lin1 = nn.Linear(self.context_size * embedding_size, self.lin_size)
        self.lin2 = nn.Linear(self.lin_size, self.vocab_sz)
        self.nonlin = nonlin

    def forward(
        self,
        x,
    ):
        assert len(x.shape) == 2 and x.shape[1] == self.context_size
        c = self.embedding(x)
        concat = c.flatten(start_dim=1, end_dim=2)

        concat = self.nonlin(concat)

        res = self.lin1(concat)
        logits = self.lin2(res)

        return logits

    def generate(self, x, num_tokens):
        assert x.shape[0] == self.context_size

        out_idx = []
        for _ in range(num_tokens):
            probs = self(x.unsqueeze(0)).squeeze().softmax(dim=-1)
            out = torch.multinomial(probs, 1)
            out_idx.append(out.item())
            x = torch.cat([x[1:self.context_size], out])

        return out_idx

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        assert len(x.shape) == 2 and x.shape[1] == self.context_size
        assert len(y.shape) == 1 and y.shape[0] == x.shape[0]

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss.item()

    def train_batch(
        self,
        x,
        y,
        optimizer,
    ):
        self.train()

        logits = self(x)

        loss = F.cross_entropy(logits, y)
        loss.backward()

        optimizer.step()

        return loss.item()


class NeuralDataset(Dataset):
    def __init__(self, txt_tensor: torch.Tensor, device="cpu", context_size=3, ):
        assert len(txt_tensor.shape) == 1, "1D tensor"
        assert txt_tensor.dtype == torch.long

        self.data = txt_tensor.to(device)
        self.context_size = context_size

    def __len__(self):
        return self.data.shape[0] - self.context_size

    def __getitem__(self, idx: int):

        x = self.data[idx:idx + self.context_size]
        y = self.data[idx + self.context_size]

        return x, y
