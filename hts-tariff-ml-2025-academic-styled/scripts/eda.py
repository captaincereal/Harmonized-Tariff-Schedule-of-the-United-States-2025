import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

INP = "data/clean_hts_2025.csv"
FIG = "figures"

def main():
    df = pd.read_csv(INP)
    os.makedirs(FIG, exist_ok=True)

    # Indent distribution
    if "Indent" in df.columns:
        counts = df["Indent"].value_counts().sort_index()
        if len(counts):
            plt.figure(figsize=(6,4))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title("Number of Items by Hierarchy Level (Indent)")
            plt.xlabel("Indent Level"); plt.ylabel("Count")
            plt.tight_layout(); plt.savefig(os.path.join(FIG, "indent_distribution.png"), dpi=180); plt.close()

    # General duty histogram
    if "General Duty (%)" in df.columns:
        gen = pd.to_numeric(df["General Duty (%)"], errors="coerce").dropna()
        if len(gen):
            plt.figure(figsize=(6,4))
            plt.hist(gen, bins=30)
            plt.title("Distribution of General Duty Rates"); plt.xlabel("Duty Rate (%)"); plt.ylabel("Count")
            plt.tight_layout(); plt.savefig(os.path.join(FIG,"general_duty_hist.png"), dpi=180); plt.close()

    print("Saved figures to", FIG)

if __name__ == "__main__":
    main()
