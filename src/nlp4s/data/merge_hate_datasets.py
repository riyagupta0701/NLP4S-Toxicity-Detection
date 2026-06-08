import io
import requests
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]  # src/nlp4s/data/
merged_dataset_output_folder = PROJECT_ROOT / "data" / "processed" / "train_split"

if __name__ == "__main__":
    #Multi3Hate dataset
    print("Loading Multi3Hate ...")
    ds_m3h = load_dataset("MinhDucBui/Multi3Hate")
    df_m3h = ds_m3h["train"].to_pandas()

    LANG_TO_COL = {
        "en": "US", 
        "de": "DE", 
        "es": "MX",
        "zh": "CN",
        "hi": "IN"
        }

    def get_m3h_label(row):
        col = LANG_TO_COL.get(row["Language"])
        val = row[col] if col else None
        return int(val) if pd.notna(val) else None
    
    df_m3h["label"] = df_m3h.apply(get_m3h_label, axis=1)
    df_m3h = df_m3h.dropna(subset=["label"])
    df_m3h["label"] = df_m3h["label"].astype(int)
    df_m3h = df_m3h.rename(columns={"Caption": "text", "Language": "language"})
    df_m3h["source"] = "multi3hate"
    df_m3h = df_m3h[["text", "label", "language", "source"]]
    
    print(f"  → {len(df_m3h):,} rows")

    #MLMA dataset

    print("Loading MLMA ...")

    ds_mlma = load_dataset("nedjmaou/MLMA_hate_speech")
    df_mlma = ds_mlma["train"].to_pandas()
    

    df_mlma["label"] = (df_mlma["sentiment"] != "normal").astype(int)
    df_mlma = df_mlma.rename(columns={"tweet": "text"})
    df_mlma["source"] = "mlma"

    #Language detection with langdetect

    try:
        from langdetect import detect, LangDetectException
        def safe_detect(text):
            try:
                return detect(str(text))
            except LangDetectException:
                return "unknown"
        print("  Detecting languages (this may take a minute) ...")
        df_mlma["language"] = df_mlma["text"].apply(safe_detect)
    except ImportError:
        print("  langdetect not installed — language set to 'unknown'.")
        print("  Run:  pip install langdetect")
        df_mlma["language"] = "unknown"
    
    df_mlma = df_mlma[["text", "label", "language", "source"]]
    print(f"  → {len(df_mlma):,} rows")

    #HASOC 2020

    print("Loading HASOC 2020 ...")

    BASE = "https://raw.githubusercontent.com/roushan-raj/HASOC-2020/master/Dataset"
    
    HASOC_FILES = {
        "en": {
            "train": f"{BASE}/Train Data/hasoc_2020_en_train.xlsx",
            "test":  f"{BASE}/Test Data/english_test_1509.csv",
        },
        "de": {
            "train": f"{BASE}/Train Data/hasoc_2020_de_train.xlsx",
            "test":  f"{BASE}/Test Data/german_test_1509.csv",
        },
        "hi": {
            "train": f"{BASE}/Train Data/hasoc_2020_hi_train.xlsx",
            "test":  f"{BASE}/Test Data/hindi_test_1509.csv",
        },
    }
    
    hasoc_frames = []
    for lang, splits in HASOC_FILES.items():
        for split_name, url in splits.items():
            url = url.replace(" ", "%20")
            if url.endswith(".xlsx"):
                response = requests.get(url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to fetch {url}\n")
                df = pd.read_excel(io.BytesIO(response.content), engine="openpyxl")
            else:
                df = pd.read_csv(url)
            #cols: tweet_id, text, task1, task2, ID
            df["language"] = lang
            df["split"] = split_name
            hasoc_frames.append(df)
    
    df_hasoc = pd.concat(hasoc_frames, ignore_index=True)
    df_hasoc["label"] = (df_hasoc["task1"] == "HOF").astype(int)
    df_hasoc["source"] = "hasoc2020"
    print(f"  → {len(df_hasoc):,} rows")

    def clean(df):
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"].str.strip().str.len() > 0]
        df = df.drop_duplicates(subset=["text"])
        df["text"] = df["text"].str.strip()
        return df.reset_index(drop=True)
    
    df_m3h   = clean(df_m3h)
    df_mlma  = clean(df_mlma)
    df_hasoc = clean(df_hasoc)

    # Strategy:
    #   - HASOC: use its own pre-defined train/val split
    #   - Multi3Hate & MLMA: stratified 90/10 random split
    
    hasoc_train = df_hasoc[df_hasoc["split"] == "train"].drop(columns="split")
    hasoc_val   = df_hasoc[df_hasoc["split"] == "test"].drop(columns="split")
    
    def stratified_split(df, val_size=0.10, seed=42):
        train, val = train_test_split(
            df, test_size=val_size, stratify=df["label"], random_state=seed
        )
        return train.reset_index(drop=True), val.reset_index(drop=True)
    
    m3h_train,  m3h_val  = stratified_split(df_m3h)
    mlma_train, mlma_val = stratified_split(df_mlma)
    
    train_df = pd.concat([hasoc_train, m3h_train, mlma_train], ignore_index=True)
    val_df   = pd.concat([hasoc_val,   m3h_val,   mlma_val],   ignore_index=True)
    
    #shuffle selection
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n── Summary ───────────────────────────────────────────────────────────")
    for name, df in [("TRAIN", train_df), ("VAL", val_df)]:
        print(f"\n{name}  ({len(df):,} rows total)")
        summary = df.groupby(["source", "language", "label"]).size().rename("count")
        print(summary.to_string())
    
    print(f"\nOverall class balance:")
    print(f"  train  label=0: {(train_df['label']==0).sum():,}  "
        f"label=1: {(train_df['label']==1).sum():,}")
    print(f"  val    label=0: {(val_df['label']==0).sum():,}  "
        f"label=1: {(val_df['label']==1).sum():,}")
    
    merged_dataset_output_folder.mkdir(parents=True, exist_ok=True) #if output folder not made
    train_df.to_csv(f"{merged_dataset_output_folder}/hate_train.csv", index=False)
    val_df.to_csv(f"{merged_dataset_output_folder}/hate_val.csv", index=False)
    print("\nSaved hate_train.csv and hate_val.csv")