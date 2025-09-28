# Optional: SBERT embeddings + Logistic Regression for 3-class duty bands (CPU-friendly)
import os, re, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer

INP, ART, FIG = "data/clean_hts_2025.csv", "artifacts", "figures"
os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def parse_percent(v):
    if pd.isna(v): return np.nan
    s=str(v)
    if re.search(r"\bfree\b", s, flags=re.I): return 0.0
    m=re.search(r"(-?\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else np.nan

def band(d):
    if d==0.0: return 0
    if (d>0.0) and (d<=5.0): return 1
    if d>5.0: return 2
    return None

df = pd.read_csv(INP)
if "General Duty (%)" not in df.columns or df["General Duty (%)"].isna().all():
    df["General Duty (%)"] = df["General Rate of Duty"].apply(parse_percent)
df["desc_clean"] = df["Description"].astype(str).str.lower().str.replace(r"[^a-z]+"," ", regex=True).str.strip()

usable = (df["desc_clean"].str.len()>0) & df["General Duty (%)"].notna()
data = df.loc[usable, ["desc_clean","General Duty (%)"]].copy()
data["label"] = data["General Duty (%)"].apply(band)
data = data.dropna(subset=["label"]).reset_index(drop=True).astype({"label":int})

tr, tmp = train_test_split(data, test_size=0.30, stratify=data["label"], random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=42)

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed(texts, bs=256): return sbert.encode(list(texts), batch_size=bs, convert_to_numpy=True, show_progress_bar=True)

Xtr, Xva, Xte = embed(tr["desc_clean"]), embed(va["desc_clean"]), embed(te["desc_clean"])
ytr, yva, yte = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()

clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", class_weight="balanced", max_iter=1000, n_jobs=-1)
clf.fit(Xtr, ytr)

def eval_split(X,y):
    yp = clf.predict(X)
    return accuracy_score(y,yp), f1_score(y,yp,average="macro"), f1_score(y,yp,average="weighted"), confusion_matrix(y,yp,labels=[0,1,2])

va_acc, va_f1m, va_f1w, _ = eval_split(Xva,yva)
te_acc, te_f1m, te_f1w, cm = eval_split(Xte,yte)

import matplotlib.pyplot as plt, numpy as np
plt.figure(figsize=(5,4)); plt.imshow(cm, interpolation="nearest"); plt.title("SBERT + LR Duty Bands - CM")
plt.colorbar(); ticks=np.arange(3); plt.xticks(ticks,["Free","1–5%"," >5%"]); plt.yticks(ticks,["Free","1–5%"," >5%"])
th=cm.max()/2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center",color=("white" if cm[i,j]>th else "black"))
plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout(); plt.savefig(os.path.join(FIG,"bands_fast_confusion.png"), dpi=180); plt.close()

with open(os.path.join(ART,"bands_fast_metrics.json"),"w") as f:
    json.dump({"val_accuracy":float(va_acc),"val_f1_macro":float(va_f1m),"val_f1_weighted":float(va_f1w),
               "test_accuracy":float(te_acc),"test_f1_macro":float(te_f1m),"test_f1_weighted":float(te_f1w)}, f, indent=2)
print("Saved artifacts and figures.")
