import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Paths (keep in sync with transformer_train_predict.py)
DATA_CSV = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/daily_features_for_transformer.csv"
META_JSON = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/daily_features_for_transformer.meta.json"
CKPT_PATH = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/analysis_output/transformer_model.pth"
PLOT_PATH = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/analysis_output/transformer_pred.png"

# 优先用 GPU，其次 Apple MPS，最后 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Hyperparams (must match training)
SEQ_LEN = 60
PRED_LEN = 1
BATCH_SIZE = 64
TRAIN_RATIO = 0.7


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

    y_log = np.log1p(y).reshape(-1, 1)
    return X, y_log, dates


def predict_and_plot():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    X, y_log, dates = load_data()

    # PyTorch 2.6 默认 weights_only=True，这里需要加载 scaler（pickle 对象），故显式设为 False
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    cfg = ckpt["config"]
    input_dim = cfg["input_dim"]

    X_scaled = scaler.transform(X)

    n = len(X_scaled)
    train_size = int(n * TRAIN_RATIO)
    X_test = X_scaled[train_size:]
    y_test = y_log[train_size:]

    test_ds = TimeSeriesDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CrimeTransformer(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    preds, acts = [], []
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

    tail = min(180, len(preds_lin))
    plot_dates = aligned_dates[-tail:]
    plot_actual = acts_lin[-tail:]
    plot_preds = preds_lin[-tail:]

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(plot_dates, plot_actual, label="Actual", linewidth=1.2)
    plt.plot(plot_dates, plot_preds, label="Pred", linewidth=1.2, alpha=0.8)
    plt.title("Transformer Forecast (Test Tail)")
    plt.xlabel("Date")
    plt.ylabel("Crime Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    print(f"Saved plot to {PLOT_PATH}")
    preview = min(10, len(plot_dates))
    print("Recent samples (actual vs pred):")
    for d, a, p in list(zip(plot_dates, plot_actual, plot_preds))[-preview:]:
        print(f"{d}: actual={a:.1f}, pred={p:.1f}")


if __name__ == "__main__":
    predict_and_plot()
