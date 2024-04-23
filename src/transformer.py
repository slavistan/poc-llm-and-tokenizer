import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import os

class SelfAttentionBlock(nn.Module):
    def __init__(self, modelConfig):
        super().__init__()
        self.modelConfig = modelConfig
        embeddingSize = self.modelConfig["embeddingSize"]
        blockSize = self.modelConfig["blockSize"]
        self.dim = np.sqrt(embeddingSize)

        self.linQ = nn.Linear(embeddingSize, embeddingSize, bias=False)
        self.linK = nn.Linear(embeddingSize, embeddingSize, bias=False)
        self.linV = nn.Linear(embeddingSize, embeddingSize, bias=False)

        self.triu = (
            torch.triu(torch.ones((blockSize, blockSize)), diagonal=1).to(self.modelConfig["device"]) == 1
        )

    def forward(self, x):
        Q = self.linQ(x)
        V = self.linV(x)
        K = self.linK(x)

        B, T, C = x.shape

        K = torch.transpose(K, 1, 2)
        weights = (Q @ K) / self.dim
        weights = weights.masked_fill(self.triu[:T, :T], -torch.inf)
        weights = nn.functional.softmax(weights, -1)
        return weights @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, modelConfig):
        super().__init__()
        self.modelConfig = modelConfig
        self.heads = nn.ModuleList(
            [SelfAttentionBlock(self.modelConfig) for _ in range(self.modelConfig["headSize"])]
        )
        self.lin = nn.Linear(self.modelConfig["embeddingSize"] * self.modelConfig["headSize"], self.modelConfig["embeddingSize"])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.lin(out)

class Linear(nn.Module):

    def __init__(self, modelConfig):
        super().__init__()
        self.modelConfig = modelConfig
        self.net = nn.Sequential(
            nn.Linear(self.modelConfig["embeddingSize"], self.modelConfig["embeddingSize"] * self.modelConfig["linScale"]),
            nn.ReLU(),
            nn.Linear(self.modelConfig["embeddingSize"] * self.modelConfig["linScale"], self.modelConfig["embeddingSize"]),
            nn.Dropout(self.modelConfig["dropout"]),
        )

    def forward(self, x):
        return self.net(x)


class TransformerDecoder(nn.Module):
    def __init__(self, modelConfig):
        super().__init__()
        self.modelConfig = modelConfig

        self.layers = nn.ModuleList(
            [
                nn.LayerNorm(self.modelConfig["embeddingSize"]),
                MultiHeadSelfAttention(self.modelConfig),
                nn.LayerNorm(self.modelConfig["embeddingSize"]),
                Linear(self.modelConfig),
            ]
        )

        for i in range(self.modelConfig["numLayers"] - 1):
            self.layers.extend(
                [
                    nn.LayerNorm(self.modelConfig["embeddingSize"]),
                    MultiHeadSelfAttention(self.modelConfig),
                    nn.LayerNorm(self.modelConfig["embeddingSize"]),
                    Linear(self.modelConfig),
                ]
            )

        self.ln = nn.LayerNorm(self.modelConfig["embeddingSize"])
        self.final_linear = nn.Linear(self.modelConfig["embeddingSize"], self.modelConfig["vocabSize"])

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, MultiHeadSelfAttention):
                x = x + layer(x)
            if isinstance(layer, Linear):
                x = x + layer(x)
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)

        return self.final_linear(self.ln(x))

class Transformer(nn.Module):

    def __init__(self, modelConfig, encode, decode):
        super().__init__()
        self.modelConfig = modelConfig
        self.encode = encode
        self.decode = decode
        self.decoder = TransformerDecoder(self.modelConfig)
        self.embed = nn.Embedding(self.modelConfig["vocabSize"], self.modelConfig["embeddingSize"])
        self.positional_encoding = nn.Embedding(self.modelConfig["blockSize"], self.modelConfig["embeddingSize"])

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x)
        pos_embed = self.positional_encoding(
            torch.arange(T, device=torch.device(self.modelConfig["device"]))
        )
        x = x + pos_embed
        pred = self.decoder(x)
        return pred

    def generate(self, context, max_new_tokens, sample=True, finish_sentence=False):

        idx = torch.tensor([self.encode(context)]).to(self.modelConfig["device"])

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.modelConfig["blockSize"]:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if sample:
                res = torch.multinomial(probs, 1)
            else:
                res = torch.argmax(probs)
            idx_next = torch.tensor([[res]]).to(self.modelConfig["device"])
            idx = torch.cat((idx, idx_next), dim=1)
            if finish_sentence and res == torch.tensor(self.encode(["\n"])).to(self.modelConfig["device"]):
                break

        out = self.decode(idx[0].tolist())

        return out


# TODO: Add checkpoints and config file
class MakeTransformer():
    def __init__(self, transformerClass, textCorpus, numLayers, embeddingSize, headSize, blockSize, linScale=5, dropout=0.2, maxIters=500, learningRate=0.0005, batchSize=64, evalInterval=500, evalIters=200, checkpointPath="./transformer-checkpoints", checkpointName="default"):
        self.textCorpus = textCorpus
        chars = sorted(list(set(self.textCorpus)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: "".join([self.itos[i] for i in l])
        vocabSize = len(self.itos)
        data = torch.tensor(self.encode(self.textCorpus), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        modelPath = f"{checkpointPath}/{checkpointName}"
        os.makedirs(modelPath, exist_ok=True)

        self.modelConfig = {
            "numLayers": numLayers,
            "embeddingSize": embeddingSize,
            "headSize": headSize,
            "blockSize": blockSize,
            "linScale": linScale,
            "dropout": dropout,
            "vocabSize": vocabSize,
            "device": device
        }

        self.trainConfig = {
            "maxIters": maxIters,
            "learningRate": learningRate,
            "batchSize": batchSize,
            "evalInterval": evalInterval,
            "evalIters": evalIters,
            "checkpointPath": checkpointPath,
            "checkpointName": checkpointName,
            "modelPath": modelPath
        }

        self.model = transformerClass(self.modelConfig, self.encode, self.decode).to(self.modelConfig["device"])

    def print_model_states(self):
        print("Available Model states:")
        print("----")
        files = os.listdir(self.trainConfig["modelPath"])
        for fileName in files:
            if fileName.endswith(".pth"):
                print(fileName.split("_")[-1].split(".")[0])
        print("----")

    # TODO: Leads to training still starting at iteration number 0 and overwriting model states
    def load_model(self, iteration):
        self.model.load_state_dict(torch.load(f"{self.trainConfig['modelPath']}/transformer_{iteration}.pth"))

    def get_batch(self, split, batch_size):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.modelConfig["blockSize"], (batch_size,))
        x = torch.stack([data[i : i + self.modelConfig["blockSize"]] for i in ix])
        y = torch.stack([data[i + 1 : i + self.modelConfig["blockSize"] + 1] for i in ix])
        x, y = x.to(self.modelConfig["device"]), y.to(self.modelConfig["device"])
        return x, y


    def get_loss(self, pred, target):
        B, T, C = pred.shape
        pred = pred.view(B * T, C)
        target = target.view(B * T)
        return F.cross_entropy(pred, target)


    @torch.no_grad()
    def estimate_loss(self, model, eval_iters, batch_size):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split, batch_size)
                pred = model(X)
                loss = self.get_loss(pred, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    def train_model(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainConfig["learningRate"])

        for iter in range(self.trainConfig["maxIters"]):
            with torch.autocast(device_type=self.modelConfig["device"], dtype=torch.float16):

                if iter % self.trainConfig["evalInterval"] == 0 or iter == self.trainConfig["maxIters"] - 1:
                    losses = self.estimate_loss(self.model, self.trainConfig["evalIters"], self.trainConfig["batchSize"])
                    print(
                        f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )

                    # torch.save(self.model.state_dict(), f"{self.trainConfig['modelPath']}/transformer_{iter}.pth")


                xb, yb = self.get_batch("train", self.trainConfig["batchSize"])
                pred = self.model(xb)
                loss = self.get_loss(pred, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        return losses["val"]
