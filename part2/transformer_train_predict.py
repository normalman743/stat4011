import os
import json
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATA_CSV = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/daily_features_for_transformer.csv"
META_JSON = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/daily_features_for_transformer.meta.json"
PLOT_PATH = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/analysis_output/transformer_pred.png"
CKPT_PATH = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/analysis_output/transformer_model.pth"

SEQ_LEN = 60
PRED_LEN = 1
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
TRAIN_RATIO = 0.7

# 设备选择
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class TimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.x = torch.FloatTensor(features)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.x) - SEQ_LEN - PRED_LEN + 1

    def __getitem__(self, idx):
        x = self.x[idx : idx + SEQ_LEN]
        y = self.y[idx + SEQ_LEN : idx + SEQ_LEN + PRED_LEN].reshape(-1)
        return x, y


class CrimeTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_emb[:, : x.size(1), :]
        x = self.encoder(x)
        out = x[:, -1, :]
        return self.head(out)


def load_data():
    meta = json.load(open(META_JSON, "r", encoding="utf-8"))
    df = pd.read_csv(DATA_CSV)
    df[meta["date_col"]] = pd.to_datetime(df[meta["date_col"]])
    df = df.sort_values(meta["date_col"])

    dates = df[meta["date_col"]].values
    y = df[meta["count_col"]].values.astype(np.float32)
    X = df.drop(columns=[meta["date_col"], meta["count_col"]]).values.astype(np.float32)

    # log1p 目标，稳定训练
    y_log = np.log1p(y).reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_log, dates, scaler


def train_and_predict():
    X, y, dates, scaler = load_data()
    n = len(X)
    train_size = int(n * TRAIN_RATIO)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds = TimeSeriesDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CrimeTransformer(input_dim=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}, Device: {DEVICE}")
    losses = []
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(-1)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(train_dl))
        losses.append(avg)
        if (ep + 1) % 2 == 0:
            print(f"Epoch {ep+1}/{EPOCHS} loss={avg:.4f}")

    # 预测
    model.eval()
    preds = []
    acts = []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()
            preds.extend(out)
            acts.extend(yb.unsqueeze(-1).numpy())

    preds = np.array(preds).reshape(-1)
    acts = np.array(acts).reshape(-1)
    preds_lin = np.expm1(preds)
    acts_lin = np.expm1(acts)

    start_idx = train_size + SEQ_LEN
    aligned_dates = dates[start_idx : start_idx + len(preds_lin)]

    mae = np.mean(np.abs(preds_lin - acts_lin))
    print(f"Test MAE (original scale): {mae:.2f}")

    tail = min(180, len(preds_lin))
    plt.figure(figsize=(10, 4))
    plt.plot(aligned_dates[-tail:], acts_lin[-tail:], label="Actual", linewidth=1.2)
    plt.plot(aligned_dates[-tail:], preds_lin[-tail:], label="Pred", linewidth=1.2, alpha=0.8)
    plt.title("Transformer Forecast (Test Tail)")
    plt.xlabel("Date")
    plt.ylabel("Crime Count")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Saved plot to {PLOT_PATH}")

    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler,
        "config": {
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "input_dim": X.shape[1],
            "train_ratio": TRAIN_RATIO,
            "target_transform": "log1p",
        },
    }, CKPT_PATH)
    print(f"Saved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    train_and_predict()
