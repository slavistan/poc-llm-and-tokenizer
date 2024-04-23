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
        context_size = 3
    ):
        super().__init__()

        self.vocab_sz = vocab_sz
        self.embedding_size = embedding_size
        self.lin_size = lin_size
        self.context_size = context_size
        self.embedding = nn.Embedding(self.vocab_sz, self.embedding_size)
        self.lin1 = nn.Linear(self.context_size * embedding_size, self.lin_size)
        self.lin2 = nn.Linear(self.lin_size, self.vocab_sz)

    def forward(
        self,
        x, # [idx_1, idx_2, idx_3, ..., idx_context_size]
    ):
        #c1 = self.embedding(x[0])
        #c2 = self.embedding(x[1])
        #c3 = self.embedding(x[2])

        c = self.embedding(x)

        #concat = torch.cat((c1, c2, c3))
        concat = c.flatten()

        concat = nn.functional.tanh(concat)

        res = self.lin1(concat)
        logits = self.lin2(res)

        return logits

    def generate(self, x, num_tokens):
        out_idx = []
        for _ in range(num_tokens):
            probs = self.embedding(x).squeeze().softmax(dim=-1)
            out = torch.multinomial(probs, 1)
            out_idx.append(out.item())
            x = out

        return out_idx


    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
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

        y = self.data[idx + self.context_size + 1]

        return x, y
