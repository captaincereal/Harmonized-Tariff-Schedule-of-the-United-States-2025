import os, re, pandas as pd, numpy as np

RAW = "data/raw/hts_2025.csv"     # put your raw file here
OUT = "data/clean_hts_2025.csv"

def parse_percent(value):
    if pd.isna(value): return np.nan
    s = str(value)
    if re.search(r"\bfree\b", s, flags=re.I): return 0.0
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else np.nan

def extract_codes(value):
    if pd.isna(value): return []
    s = str(value)
    paren = re.findall(r"\(([^)]+)\)", s)
    for g in paren:
        parts = [p.strip() for p in g.split(",") if p.strip()]
        if parts: return parts
    return []

def main():
    if not os.path.exists(RAW):
        raise SystemExit(f"Raw CSV not found: {RAW}")
    df = pd.read_csv(RAW)
    df.columns = [c.strip() for c in df.columns]

    if "HTS Number" in df.columns:
        df["HTS Number"] = df["HTS Number"].ffill()
        df["Chapter"] = df["HTS Number"].astype(str).str[:2]

    if "General Duty (%)" not in df.columns or df["General Duty (%)"].isna().all():
        df["General Duty (%)"] = df["General Rate of Duty"].apply(parse_percent)
    if "Column 2 Duty (%)" not in df.columns and "Column 2 Rate of Duty" in df.columns:
        df["Column 2 Duty (%)"] = df["Column 2 Rate of Duty"].apply(parse_percent)

    if "Special Rate Codes" not in df.columns and "Special Rate of Duty" in df.columns:
        df["Special Rate Codes"] = df["Special Rate of Duty"].apply(extract_codes)

    df["desc_clean"] = (
        df["Description"].astype(str)
          .str.lower()
          .str.replace(r"[^a-z]+", " ", regex=True)
          .str.strip()
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
