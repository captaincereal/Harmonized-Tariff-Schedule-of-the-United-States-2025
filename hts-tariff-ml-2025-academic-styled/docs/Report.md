# HTS 2025 — Academic Report

## 1. Introduction
We evaluate whether unstructured product descriptions in the U.S. Harmonized Tariff Schedule (HTS 2025) predict tariff outcomes. We aim for a transparent, CPU‑friendly pipeline suitable for an academic portfolio.

## 2. Data & Preprocessing
- Convert duty text to numeric % (treat "Free" as 0.0).
- Forward‑fill hierarchical codes; derive Chapter.
- Extract special‑rate codes from parentheses.
- Clean description text.

## 3. Methods
- EDA (indent distribution; duty histograms; chapter summaries).
- Binary classification (Free vs Non‑Free) using TF‑IDF + Logistic Regression/XGBoost.
- Optional 3‑class duty bands with sentence embeddings (MiniLM) + multinomial LR.
- Metrics: accuracy, F1, ROC/PR (binary).

## 4. Results (from the provided run)
- Logistic Regression — Acc 0.745 · F1 0.697 · ROC‑AUC 0.813
- XGBoost — Acc 0.724 · F1 0.657 · ROC‑AUC 0.796

## 5. Discussion
Language carries useful signal (AUC ~0.8) for Free vs Non‑Free, helping triage items for review. Boundaries near low duties remain challenging; contextual embeddings or structured attributes could help.

## 6. Limitations
Educational only; text‑only; class imbalance; domain rules omitted.

## 7. Conclusion
A reproducible, light‑weight text pipeline offers practical screening value and is a strong portfolio piece; future work can add richer features and calibration.


> [!NOTE]
> This report emphasizes transparent baselines and documented assumptions to meet academic review standards.
