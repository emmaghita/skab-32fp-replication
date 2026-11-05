import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from core.dataset import get_data
from core.model import TinyTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None, help="checkpoint path (default: checkpoints/fp32/seed<seed>.pt)")
args = parser.parse_args()
out = args.out or f"checkpoints/fp32/seed{args.seed}.pt"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

torch.manual_seed(args.seed)

# settings taken from the paper
D_MODEL = 24
BATCH = 96
LR = 5.064e-3
EPOCHS = 100
PATIENCE = 10

train, valid, test, mean, std = get_data()
X_tr, y_tr = torch.from_numpy(np.concatenate(train[0])), torch.from_numpy(np.concatenate(train[1]))
X_va, y_va = torch.from_numpy(np.concatenate(valid[0])), torch.from_numpy(np.concatenate(valid[1]))

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True, drop_last=True)
valid_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH, shuffle=False)

model = TinyTransformer(num_in=8, num_out=1, d_model=D_MODEL)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

best_val, bad_epochs = float("inf"), 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()       # clear old gradients
        loss.backward()             # compute how each weight should change
        optimizer.step()            # change the weights a little
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in valid_loader:
            val_loss += loss_fn(model(X), y).item() * len(X)   # weight by batch size (last batch is smaller)
    val_loss /= len(valid_loader.dataset)

    print(f"epoch {epoch+1:3d} | train {train_loss:.4f} | valid {val_loss:.4f}")

   # early stoping
    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), out)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"early stopping at epoch {epoch+1}")
            break

print(f"best valid loss: {best_val:.4f} -> saved to {out}")