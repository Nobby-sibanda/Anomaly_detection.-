# Anomaly Detection in Financial Transactions

**Author:** Nobukhosi Sibanda  
**Methods:** Gaussian Mixture Model (GMM) + Isolation Forest (IF)  
**Type:** Practical Data Science Project — Unsupervised Anomaly Detection

---

## Overview

This project detects anomalous (potentially fraudulent) transactions in a financial dataset without using any fraud labels. Both models are trained exclusively on normal transaction behaviour and flag transactions that deviate significantly from that learned baseline.

```
financial_anomaly_data.xlsx
        │
        ▼
Feature Engineering (10 core + one-hot encoded features)
        │
        ├──► Gaussian Mixture Model (GMM)    ──► NLL anomaly score
        └──► Isolation Forest (IF)           ──► Path-length anomaly score
                                                      │
                                              Threshold @ 99.5th percentile
                                                      │
                                    Evaluation: ROC-AUC, PR-AUC, Recall, Confusion Matrix
```

---

## Dataset

**File:** `financial_anomaly_data.xlsx`

| Property | Value |
|----------|-------|
| Columns | Timestamp, AccountID, Amount, Merchant, TransactionType, Location |
| Unique AccountIDs | 15 |
| Unique Merchants | 10 |
| Transaction Types | 3 |
| Locations | 5 (New York, San Francisco, Los Angeles, London, Tokyo) |
| Amount range | $10.51 – $978,942 |
| Amount skewness | ≈ 0.4 (mild right skew) |
| Amount kurtosis | ≈ 9.7 (heavy tails driven by extreme outliers) |
| Missing values | None |
| Duplicate rows | None |

**Synthetic anomaly labels** are created by flagging the top 0.5% of transactions by amount (contamination = 0.005), used solely for evaluation — not for model training.

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `LogAmount` | `log(1 + Amount)` — reduces right skew so the raw dollar scale does not dominate |
| `Hour` | Hour of transaction (0–23) |
| `DayOfWeek` | Day of week (0 = Monday) |
| `IsWeekend` | Binary flag for Saturday/Sunday |
| `IsNight` | Binary flag for 22:00–05:00 transactions |
| `ZScore` | Account-relative z-score: `(Amount − AccountMean) / AccountStd` |
| `AbsZScore` | Absolute value of ZScore |
| `MerchDev` | Deviation of Amount from the merchant's average transaction value |
| `VelocityFlag` | Impossible-travel flag — uses the Haversine formula to detect sequential transactions that would require > 900 km/h travel speed between cities |
| `LocationRiskScore` | Per-city fraction of anomalous transactions |
| One-hot columns | TransactionType (3), Location (5), Merchant (10) |

> **Data quality note:** VelocityFlag flagged ~79% of transactions as impossible travel. This is an artefact of the synthetic dataset assigning locations randomly rather than simulating realistic spatial behaviour. The feature is included but interpreted with this caveat.

---

## Data Split & Preprocessing

| Stage | Details |
|-------|---------|
| Split strategy | **Temporal** — first 70% of normal transactions → training; remaining normals + all labelled anomalies → test |
| Rationale | Prevents future data from leaking into training; mirrors real-world deployment |
| Scaling | `StandardScaler` fitted on training data only; applied to both sets |

---

## Models

### Gaussian Mixture Model (GMM)

- **Type:** Probabilistic — fits a mixture of k multivariate Gaussian components to the training data
- **Anomaly score:** Negative log-likelihood (NLL); higher NLL = lower probability under the fitted normal distribution
- **Component selection:** k chosen by minimising BIC over k ∈ {1, …, 8} (BIC penalises complexity more aggressively than AIC to avoid over-fitting)
- **Threshold:** 99.5th percentile of training NLL scores (matches 0.5% contamination assumption)

### Isolation Forest (IF)

- **Type:** Ensemble tree-based — isolates observations by randomly partitioning feature space
- **Anomaly score:** Negative average path length; shorter path = easier to isolate = more anomalous
- **Configuration:** 300 estimators, full feature set, contamination = 0.005, all CPU cores
- **Threshold:** 99.5th percentile of training scores
- **Feature attribution:** Permutation importance computed on PR-AUC drop

---

## Results

| Metric | GMM | Isolation Forest |
|--------|-----|-----------------|
| **ROC-AUC** | 0.7007 | **0.8314** |
| **PR-AUC** | 0.0394 | **0.0554** |
| Contamination | 0.5% | 0.5% |

> **Isolation Forest outperformed GMM on every metric.** It handles heavy-tailed and non-Gaussian transaction patterns more effectively because it makes no distributional assumption about the normal class.

**Top features by IF permutation importance:**  
Temporal signals (IsNight, DayOfWeek, Hour, IsWeekend) contributed strongly, alongside amount-based features (LogAmount, ZScore, AbsZScore).

**Model agreement analysis:**  
- Transactions flagged by **both** models represent the highest-confidence anomalies
- Isolation Forest-only flags tend to capture temporal outliers
- GMM-only flags tend to capture distribution-based outliers

---

## Visualizations

The notebook produces the following charts:

| # | Figure | Description |
|---|--------|-------------|
| 1 | Amount Distribution | Raw amount + log-amount histograms with anomaly overlay and threshold line |
| 2 | Descriptive Statistics Heatmap | Mean, median, std, skewness, kurtosis for key features |
| 3 | Spearman Correlation Matrix | Rank-order correlations between engineered features |
| 4 | LogAmount vs Z-Score Scatter | Anomalies cluster in the high-amount, high-Z-score corner |
| 5 | Feature Boxplots | Normal vs Anomaly distributions for LogAmount, AbsZScore, MerchDev |
| 6 | TransactionType & Location Breakdown | Count of normal vs anomaly per category |
| 7 | Geospatial 4-Panel Dashboard | City bubble map, velocity flag counts, city-pair travel heatmap, per-account flags |
| 8 | BIC/AIC Curve | Component selection for GMM |
| 9 | ROC Curves | GMM vs Isolation Forest across all thresholds |
| 10 | Precision-Recall Curves | Performance at extreme class imbalance (0.5% anomalies) |
| 11 | Confusion Matrices | True/false positives and negatives at contamination threshold |
| 12 | Comparative Metrics Bar Chart | ROC-AUC, PR-AUC, Recall side by side |
| 13 | PCA 2D Projections | Test set projected onto PC1/PC2, coloured by GMM score, IF score, and true labels |
| 14 | Permutation Feature Importance | Top-15 features ranked by PR-AUC drop (Isolation Forest) |
| 15 | Top-20 Anomalies Heatmap | Column-normalised view of the 20 highest-scored true anomalies |

---

## Contamination Rate Tuning

Section 3.7 sweeps contamination rates `[0.5%, 1%, 2%, 3%, 5%]` for both models, reporting precision, recall, F1, ROC-AUC, and PR-AUC at each rate. This demonstrates the precision-recall trade-off and helps choose a threshold appropriate for a given operational cost of false positives vs false negatives.

---

## Key Findings

1. **Isolation Forest is the better model** for this dataset. Its non-parametric, tree-based isolation mechanism handles the heavy tails and irregular structure of financial transaction data more effectively than the Gaussian assumption underlying GMM.
2. **LogAmount and account-relative z-scores are the most discriminating features** — anomalies cluster in the high-amount, high-ZScore region of the feature space.
3. **Temporal features matter** — IsNight, DayOfWeek, and Hour ranked highly in IF permutation importance, indicating that unusual timing is a strong fraud signal even independently of amount.
4. **Low recall at contamination threshold** — with only 0.5% anomalies, both models are best used for ranking and comparison rather than binary deployment decisions without threshold tuning.
5. **Synthetic dataset limitation** — VelocityFlag's 79% flag rate reveals that location data was assigned randomly in the synthetic source, reducing its practical signal value in this experiment.

---

## Notebook Structure

| Section | Content |
|---------|---------|
| 3.1 | Data preparation and initial inspection |
| 3.2a–e | Feature engineering (temporal, amount, geospatial, encoding, label creation) |
| 3.3 | EDA visualizations (7 figures) |
| 3.3a–d | Model fitting — GMM component selection, GMM training, IF training, IF scoring |
| 3.4 | Model evaluation — ROC-AUC, PR-AUC, recall, confusion matrices |
| 3.5 | PCA projection and permutation feature importance |
| 3.6 | Model overlap and agreement analysis |
| 3.7 | Contamination rate sensitivity tuning |
| Summary | Results table and key insights |
| Final Remarks | Conclusions and limitations |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl
```

---

## Usage

1. Place `financial_anomaly_data.xlsx` in the same directory as the notebook.
2. Open `Anomaly_detection..ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells in order (`Kernel → Restart & Run All`).

---

## Author

**Nobukhosi Sibanda** — [GitHub](https://github.com/Nobby-sibanda)
