import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class BigramLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_sz,
    ):
        super().__init__()

        self.vocab_sz = vocab_sz
        self.lut = nn.Embedding(self.vocab_sz, self.vocab_sz)

    def forward(
        self,
        x,
    ):
        result = self.lut(x)  # return logits
        return result

    def generate(self, x, num_tokens):
        out_idx = []
        for _ in range(num_tokens):
            probs = self.lut(x).squeeze().softmax(dim=-1)
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
        assert len(x.shape) == 1 and len(y.shape) == 1 and y.shape[0] == x.shape[0]

        self.train()

        logits = self(x)

        loss = F.cross_entropy(logits, y)
        loss.backward()

        optimizer.step()

        return loss.item()


class BigramDataset(Dataset):
    def __init__(self, txt_tensor: torch.Tensor, device="cpu"):
        assert len(txt_tensor.shape) == 1, "1D tensor"
        assert txt_tensor.dtype == torch.long

        self.data = txt_tensor.to(device)

    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, idx: int):

        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y
