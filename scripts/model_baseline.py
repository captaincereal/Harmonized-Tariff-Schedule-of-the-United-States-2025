import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier

INP = "data/clean_hts_2025.csv"
FIG = "figures"
ART = "artifacts"

def plot_cm(cm, title, path):
    import numpy as np, matplotlib.pyplot as plt
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Non-Free","Free"]); plt.yticks(ticks, ["Non-Free","Free"])
    th = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color=("white" if cm[i,j] > th else "black"))
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()

def plot_roc(ytrue,yprob,title,path):
    import matplotlib.pyplot as plt
    fpr,tpr,_=roc_curve(ytrue,yprob); roc_auc=auc(fpr,tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],"--"); plt.legend()
    plt.title(title); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()
    return roc_auc

def plot_pr(ytrue,yprob,title,path):
    import matplotlib.pyplot as plt
    prec,rec,_=precision_recall_curve(ytrue,yprob)
    plt.figure(figsize=(5,4)); plt.plot(rec,prec)
    plt.title(title); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()

def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(ART, exist_ok=True)
    df = pd.read_csv(INP)

    df["desc_clean"] = (
        df["Description"].astype(str).str.lower().str.replace(r"[^a-z]+"," ",regex=True).str.strip()
    )
    usable = (df["desc_clean"].str.len()>0) & df["General Duty (%)"].notna()
    data = df.loc[usable, ["desc_clean","General Duty (%)"]].copy()
    y = (data["General Duty (%)"] == 0.0).astype(int).to_numpy()

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95)
    X = tfidf.fit_transform(data["desc_clean"])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Logistic Regression
    lr = LogisticRegression(max_iter=300, n_jobs=-1)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:,1]
    y_pred_lr = (y_prob_lr>=0.5).astype(int)
    acc_lr, f1_lr = accuracy_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    roc_lr = plot_roc(y_test,y_prob_lr,"Logistic Regression - ROC", os.path.join(FIG,"lr_roc.png"))
    plot_pr(y_test,y_prob_lr,"LogReg - PR", os.path.join(FIG,"lr_pr.png"))
    plot_cm(cm_lr,"Logistic Regression - CM", os.path.join(FIG,"lr_cm.png"))

    # XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=200, max_depth=6, learning_rate=0.2,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42
    )
    xgb.fit(X_train, y_train)
    y_prob_xgb = xgb.predict_proba(X_test)[:,1]
    y_pred_xgb = (y_prob_xgb>=0.5).astype(int)
    acc_xgb, f1_xgb = accuracy_score(y_test, y_pred_xgb), f1_score(y_test, y_pred_xgb)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    roc_xgb = plot_roc(y_test,y_prob_xgb,"XGBoost - ROC", os.path.join(FIG,"xgb_roc.png"))
    plot_pr(y_test,y_prob_xgb,"XGBoost - PR", os.path.join(FIG,"xgb_pr.png"))
    plot_cm(cm_xgb,"XGBoost - CM", os.path.join(FIG,"xgb_cm.png"))

    pd.DataFrame([
        {"model":"LogisticRegression","accuracy":acc_lr,"f1":f1_lr,"roc_auc":roc_lr},
        {"model":"XGBoost","accuracy":acc_xgb,"f1":f1_xgb,"roc_auc":roc_xgb},
    ]).to_csv(os.path.join(ART,"model_summary.csv"), index=False)

    print("Saved figures and artifacts.")

if __name__ == "__main__":
    main()
