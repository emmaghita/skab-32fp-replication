import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from core.dataset import get_data
from core.quantized_model import QuantTinyTransformer, to_integer

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bits", type=int, default=6)
parser.add_argument("--out", default=None, help="checkpoint path (default: checkpoints/qat<bits>/seed<seed>.pt)")
args = parser.parse_args()
out = args.out or f"checkpoints/qat{args.bits}/seed{args.seed}.pt"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

torch.manual_seed(args.seed)

# same settings as train.py (paper)
D_MODEL = 24
BATCH = 96
LR = 5.064e-3
EPOCHS = 100
PATIENCE = 10

train, valid, test, mean, std = get_data()
X_tr, y_tr = torch.from_numpy(np.concatenate(train[0])), torch.from_numpy(np.concatenate(train[1]))
X_va, y_va = torch.from_numpy(np.concatenate(valid[0])), torch.from_numpy(np.concatenate(valid[1]))

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True, drop_last=True)

model = QuantTinyTransformer(num_in=8, num_out=1, d_model=D_MODEL, quant_bits=args.bits)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

best_val, bad_epochs = float("inf"), 0

for epoch in range(EPOCHS):
    model.train()                   # train mode also widens the quantizer ranges
    train_loss = 0
    for X, y in train_loader:
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()                    # eval: ranges frozen, simulated quantization
    with torch.no_grad():
        val_loss = loss_fn(model(X_va), y_va).item()   # validation set is small: one batch

    print(f"epoch {epoch+1:3d} | train {train_loss:.4f} | valid (sim) {val_loss:.4f}")

    # early stoping
    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save({"config": model.config, "state_dict": model.state_dict()}, out)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"early stopping at epoch {epoch+1}")
            break

# check the saved checkpoint in integer mode (what the FPGA will compute)
ckpt = torch.load(out, weights_only=True)
best = QuantTinyTransformer(**ckpt["config"])
best.load_state_dict(ckpt["state_dict"])
best.eval()
with torch.no_grad():
    int_loss = loss_fn(to_integer(best)(X_va), y_va).item()
print(f"best valid loss: sim {best_val:.4f} | int {int_loss:.4f} -> saved to {out}")
