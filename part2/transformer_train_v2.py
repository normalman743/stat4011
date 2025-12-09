import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================= 配置 =================
DATA_PATH = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/crime_data_embedded.pkl"
SEQ_LEN = 30
PRED_LEN = 1
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
# 优先用 GPU，其次 Apple MPS，最后 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ================= 数据集 =================
class CrimeEmbeddingDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features) - SEQ_LEN - PRED_LEN + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + SEQ_LEN]
        # 返回形状 [PRED_LEN]，后续在训练循环中再 unsqueeze 成 [batch, 1]
        y = self.targets[idx + SEQ_LEN : idx + SEQ_LEN + PRED_LEN].reshape(-1)
        return x, y

# ================= 模型 =================
class CrimeTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, 500, d_model))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_emb[:, : x.size(1), :]
        x = self.transformer(x)
        out = x[:, -1, :]
        return self.decoder(out)

# ================= 数据准备 =================
def load_daily_features():
    obj = pd.read_pickle(DATA_PATH)
    df: pd.DataFrame = obj["data"]
    emb_store = obj["embeddings"]

    # 预生成每个文本列的 embedding 矩阵与维度
    emb_dims = {col: (store["embeddings"].shape[1] if store["embeddings"].size > 0 else 0) for col, store in emb_store.items()}

    # emb 列名（id 列）
    emb_id_cols = [c for c in df.columns if c.endswith("_id")]
    num_cols = [c for c in df.columns if c.endswith("_norm")]
    cat_cols = [c for c in df.columns if c.endswith("_code")]

    # 逐日聚合
    groups = df.groupby(df["DATE OCC"].dt.date)
    daily_features = []
    daily_targets = []
    dates = []

    def id_to_vec(col: str, ids: np.ndarray) -> np.ndarray:
        store = emb_store[col]
        mat = store["embeddings"]
        if mat.size == 0:
            return np.zeros((len(ids), 0), dtype=np.float32)
        ids = ids.astype(int)
        mask = ids >= 0
        safe_ids = np.clip(ids, 0, mat.shape[0] - 1, dtype=int)
        vecs = mat[safe_ids]
        vecs[~mask] = 0.0
        return vecs

    for date, g in groups:
        count = len(g)
        # 数值与分类特征取均值（分类可改为众数，这里简化）
        num_feat = g[num_cols].mean().values if num_cols else np.array([])
        cat_feat = g[cat_cols].mean().values if cat_cols else np.array([])

        emb_feat_parts = []
        for col in emb_id_cols:
            base = col.replace("_id", "")
            vecs = id_to_vec(base, g[col].values)
            if vecs.shape[1] > 0:
                emb_feat_parts.append(vecs.mean(axis=0))
        emb_feat = np.concatenate(emb_feat_parts) if emb_feat_parts else np.array([])

        feature_vector = np.concatenate([[count], num_feat, cat_feat, emb_feat]).astype(np.float32)
        daily_features.append(feature_vector)
        daily_targets.append(count)
        dates.append(date)

    features = np.vstack(daily_features)
    targets = np.array(daily_targets, dtype=np.float32).reshape(-1, 1)
    return features, targets, dates

# ================= 训练流程 =================
def train():
    features, targets, dates = load_daily_features()
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 按时间顺序 7:3 切分（前 70% 训练，后 30% 测试/预测）
    train_size = int(len(features) * 0.7)
    train_feat, test_feat = features[:train_size], features[train_size:]
    train_targ, test_targ = targets[:train_size], targets[train_size:]

    train_ds = CrimeEmbeddingDataset(train_feat, train_targ)
    test_ds = CrimeEmbeddingDataset(test_feat, test_targ)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CrimeTransformer(input_dim=features.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Start training on {DEVICE}, train samples={len(train_ds)}, test samples={len(test_ds)}")
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(-1)  # 目标对齐为 [batch, 1]
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(train_dl))
        losses.append(avg)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} loss={avg:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Train Loss")
    plt.tight_layout()
    plt.show()

    # 简单评估：对后段测试集做预测，并打印最后若干天的实际 vs 预测
    model.eval()
    with torch.no_grad():
        preds = []
        actual = []
        for x, y in test_dl:
            x = x.to(DEVICE)
            out = model(x).cpu().numpy()
            preds.extend(out)
            actual.extend(y.unsqueeze(-1).numpy())
    preds = np.array(preds).reshape(-1)
    actual = np.array(actual).reshape(-1)
    print(f"Test MAE: {np.mean(np.abs(preds - actual)):.3f}")

    # 对齐日期：测试集的第一个可预测点对应 dates[train_size + SEQ_LEN]
    start_idx = train_size + SEQ_LEN
    aligned_dates = dates[start_idx : start_idx + len(preds)]

    preview_count = min(30, len(preds))
    print("\n最近样本（实际 vs 预测）:")
    for d, a, p in list(zip(aligned_dates, actual, preds))[-preview_count:]:
        print(f"{d}: actual={a:.1f}, pred={p:.1f}")

    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler,
        "config": {
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "input_dim": features.shape[1],
        },
    }, "crime_transformer_model.pth")
    print("Model saved to crime_transformer_model.pth")


if __name__ == "__main__":
    train()
