import os
import json
import math
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# 配置
# ---------------------------
RAW_FILE = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/Crime_Data_2010_to_Present_Cleaned_merged_and_deduped_20250929_add_by_def_fill_some_v3.1.csv"
OUTPUT_DAILY_CSV = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/daily_features_for_transformer.csv"
META_JSON = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/daily_features_for_transformer.meta.json"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "snowflake-arctic-embed2:latest"

TEXT_COLS = ["Crm Cd Desc", "Premis Desc", "Weapon Desc", "Status Desc", "AREA NAME"]
CAT_COLS = ["Vict Sex", "Vict Descent"]
NUM_COLS = ["LAT", "LON", "Vict Age"]
DATE_COL = "DATE OCC"
COUNT_COL = "crime_count"

# ---------------------------
# Ollama 调用
# ---------------------------
def get_embedding(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    payload = {"model": MODEL_NAME, "prompt": text}
    resp = requests.post(OLLAMA_URL, json=payload)
    if resp.status_code != 200:
        print(f"[warn] embedding failed: {text} -> {resp.text[:120]}")
        return None
    return resp.json().get("embedding")

# ---------------------------
# 主流程
# ---------------------------
def main():
    print(f"Loading raw data: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)

    # 时间列
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    # 对齐你当前笔记本的过滤：仅保留 2023-12-31 及以前
    cutoff = pd.Timestamp("2023-12-31")
    df = df[df[DATE_COL] <= cutoff]
    df = df.sort_values(DATE_COL)

    # 简单分类编码
    print("Encoding categorical columns (label)...")
    label_encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col + "_code"] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 数值列清洗
    print("Cleaning numeric columns...")
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 文本列去重获取 embeddings
    print("Collecting unique text values and embeddings via Ollama (per column)...")
    embedding_maps = {}
    sample_dim = None
    for col in TEXT_COLS:
        uniq = df[col].dropna().astype(str).unique()
        print(f"  {col}: {len(uniq)} unique")
        col_map = {}
        for i, val in enumerate(tqdm(uniq, desc=f"embedding {col}")):
            emb = get_embedding(val)
            if emb is not None:
                col_map[val] = emb
                if sample_dim is None:
                    sample_dim = len(emb)
            else:
                # 延迟填默认维度
                pass
        embedding_maps[col] = col_map
    if sample_dim is None:
        sample_dim = 768
        print("[warn] no embedding returned; default dim=768")

    # 按天聚合，计算均值向量
    print("Aggregating daily features (mean embedding per column)...")
    df_day = []
    grouped = df.groupby(df[DATE_COL].dt.date)
    for day, g in tqdm(grouped, desc="daily agg"):
        row = {"date": pd.to_datetime(day)}
        row[COUNT_COL] = len(g)
        # numeric mean
        for col in NUM_COLS:
            row[col + "_mean"] = g[col].mean()
        # categorical code mean (近似代表分布)
        for col in CAT_COLS:
            row[col + "_code_mean"] = g[col + "_code"].mean()
        # embedding means
        for col in TEXT_COLS:
            embs = []
            for val in g[col].astype(str).values:
                emb = embedding_maps[col].get(val)
                if emb is not None:
                    embs.append(emb)
            if embs:
                arr = np.mean(np.array(embs, dtype=np.float32), axis=0)
            else:
                arr = np.zeros(sample_dim, dtype=np.float32)
            for j in range(sample_dim):
                row[f"{col}_e{j}"] = arr[j]
        df_day.append(row)

    daily_df = pd.DataFrame(df_day).sort_values("date")
    print(f"Saving daily features to CSV: {OUTPUT_DAILY_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_DAILY_CSV), exist_ok=True)
    daily_df.to_csv(OUTPUT_DAILY_CSV, index=False)

    meta = {
        "model": MODEL_NAME,
        "text_cols": TEXT_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "embedding_dim": sample_dim,
        "count_col": COUNT_COL,
        "date_col": "date",
        "source_file": RAW_FILE,
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved meta: {META_JSON}")


if __name__ == "__main__":
    try:
        requests.get("http://localhost:11434")
    except Exception:
        print("[error] Ollama not reachable at 11434; please run 'ollama serve'")
        raise SystemExit(1)
    main()
