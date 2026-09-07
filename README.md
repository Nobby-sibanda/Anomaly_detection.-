# Anomaly Detection in Financial Transactions

**Author:** Nobukhosi Sibanda  
**Methods:** Gaussian Mixture Model (GMM) + Isolation Forest (IF) + Local Outlier Factor (LOF) + Rank-Averaged Ensemble  
**Type:** Practical Data Science Project — Unsupervised Anomaly Detection

---

## Overview

This project detects anomalous (potentially fraudulent) transactions in a financial dataset without using any fraud labels. All models are trained exclusively on normal transaction behaviour and flag transactions that deviate significantly from that learned baseline.

```
financial_anomaly_data.xlsx
        │
        ▼
Feature Engineering (10 core + one-hot encoded features)
        │
        ├──► Gaussian Mixture Model (GMM)          ──► NLL anomaly score
        ├──► Isolation Forest (IF)                 ──► Path-length anomaly score
        └──► Local Outlier Factor (LOF)             ──► Local density anomaly score
                     │
                     ├──► Rank-Averaged Ensemble of all three
                     │
                              Threshold @ 99.5th percentile
                                                      │
                                    Evaluation: ROC-AUC, PR-AUC, Recall, Confusion Matrix
```

A companion notebook, [`Real_Data_Validation.ipynb`](Real_Data_Validation.ipynb), takes the
Isolation Forest approach from this project and tests it — unchanged — against a real,
independently labeled fraud dataset. See [Real-Data Validation](#real-data-validation) below.

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

### Local Outlier Factor (LOF)

- **Type:** Density-based, *local* — unlike GMM (one global distribution) and IF (global random splits), LOF compares each transaction only to its 35 nearest neighbours
- **Anomaly score:** Negative LOF score; a transaction is anomalous if it's much sparser than its own neighbourhood, not sparse relative to the whole dataset
- **Configuration:** 35 neighbours, `novelty=True` (so it can score new/test points), contamination = 0.005
- **Threshold:** 99.5th percentile of training scores

### Ensemble

- **Type:** Rank-average of the three models' anomaly scores (ranks, not raw scores, since each model's scores live on a different scale)
- **Rationale:** each model catches somewhat different anomalies (see Model Agreement Analysis below); combining them should be at least as good as the best individual model, without having to pick one in advance

---

## Results

| Metric | GMM | Isolation Forest | Local Outlier Factor | Ensemble |
|--------|-----|-------------------|-----------------------|----------|
| **ROC-AUC** | 0.7007 | **0.8314** | 0.8011 | 0.8326 |
| **PR-AUC** | 0.0394 | 0.0554 | **0.0608** | 0.0533 |
| **Recall** | 0.0092 | 0.0184 | **0.0378** | 0.0000 |
| Contamination | 0.5% | 0.5% | 0.5% | 0.5% |

> **No single model wins on every metric.** Isolation Forest has the best ROC-AUC; Local Outlier
> Factor has the best PR-AUC and recall; GMM trails on every metric. The rank-averaged Ensemble
> nudges ROC-AUC slightly higher than any individual model — but its recall collapses to 0 at this
> threshold, a genuinely useful negative result: averaging ranks improved overall *ranking*
> quality without improving *which specific transactions* cleared the cutoff. Ranking quality
> (AUC) and threshold performance (recall at a fixed cutoff) are not the same thing to optimize.

**Top features by IF permutation importance:**  
Temporal signals (IsNight, DayOfWeek, Hour, IsWeekend) contributed strongly, alongside amount-based features (LogAmount, ZScore, AbsZScore).

**Model agreement analysis:**  
- Transactions flagged by **both** GMM and IF represent the highest-confidence anomalies
- Isolation Forest-only flags tend to capture temporal outliers
- GMM-only flags tend to capture distribution-based outliers
- LOF's local, density-based view catches a different — and here, larger — share of true anomalies than either global method, which is why it's the strongest single contributor to recall

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

1. **No single model dominates.** GMM's Gaussian assumption is the weakest fit for this heavy-tailed data (trails on every metric); Isolation Forest and LOF — two very different non-parametric approaches — are both stronger, and which one "wins" depends on whether ranking quality (IF) or recall at a fixed threshold (LOF) matters more for the use case.
2. **Combining models isn't automatically better.** The rank-averaged ensemble improved ROC-AUC slightly but its recall dropped to zero at the chosen threshold — a reminder to check an ensemble on the same metrics as every individual model, not assume combination helps.
3. **LogAmount and account-relative z-scores are the most discriminating features** — anomalies cluster in the high-amount, high-ZScore region of the feature space.
4. **Temporal features matter** — IsNight, DayOfWeek, and Hour ranked highly in IF permutation importance, indicating that unusual timing is a strong fraud signal even independently of amount.
5. **Low recall at contamination threshold** — with only 0.5% synthetic anomalies, all four approaches are best used for ranking and comparison rather than binary deployment decisions without threshold tuning. [`Real_Data_Validation.ipynb`](Real_Data_Validation.ipynb) checks this concern directly against real fraud labels — see below.
6. **Synthetic dataset limitation** — VelocityFlag's 79% flag rate reveals that location data was assigned randomly in the synthetic source, reducing its practical signal value in this experiment.

---

## Real-Data Validation

Every result above is evaluated against **self-generated labels** (top 0.5% of transactions by
amount = "anomaly") — useful for comparing models against each other, but it doesn't prove the
approach catches real fraud, since "large amount" is a rule the models could trivially learn.

[`Real_Data_Validation.ipynb`](Real_Data_Validation.ipynb) closes that gap: it takes the
Isolation Forest approach from this project — unchanged, same hyperparameters — and tests it
against the [ULB Credit Card Fraud Detection dataset](https://www.openml.org/search?type=data&status=active&id=1597)
(via `sklearn.datasets.fetch_openml`, no manual download needed): 284,807 real transactions with
**492 genuine, independently verified frauds (0.173%)**.

| Metric | Synthetic Labels (this project) | Real Fraud Labels (validation notebook) |
|--------|----------------------------------|-------------------------------------------|
| ROC-AUC | 0.8314 | **0.9507** |
| PR-AUC | 0.0554 | **0.3419** |
| Recall | 0.0184 | **0.3110** |
| Precision | — | 0.54 |

**The real numbers came out better than the synthetic evaluation, not worse.** On real,
independently labeled fraud, the same unsupervised Isolation Forest — trained with no fraud
labels at all — catches **31% of all fraud with 54% precision**, on a class that's only 0.17% of
the data. That's the direct answer to "does this actually work," and it confirms the synthetic
evaluation above wasn't overstating the approach.

One caveat: this dataset's features are pre-anonymized PCA components with no timestamp, so the
specific engineered features here (MerchDev, VelocityFlag, LocationRiskScore, hour-of-day) — and
the temporal train/test split — couldn't be reused or tested on it. What's validated is the
*modeling approach*, not the *specific feature engineering*.

---

## Notebook Structure

### `Anomaly_detection..ipynb` (main project)

| Section | Content |
|---------|---------|
| 3.1 | Data preparation and initial inspection |
| 3.2a–e | Feature engineering (temporal, amount, geospatial, encoding, label creation) |
| 3.3 | EDA visualizations (7 figures) |
| 3.3a–e | Model fitting — GMM component selection & training, IF training & scoring, LOF training & scoring |
| 3.4 | Model evaluation — ROC-AUC, PR-AUC, recall, confusion matrices (all four: GMM, IF, LOF, Ensemble) |
| 3.5 | PCA projection and permutation feature importance |
| Summary | Dynamic results table (auto-extends to however many models are registered) |
| 3.6 | Model overlap and agreement analysis (GMM vs IF) |
| Ensemble | Rank-averaged combination of GMM + IF + LOF scores |
| 3.7 | Contamination rate sensitivity tuning |
| Model Comparison | Results table and key insights across all four approaches |
| Final Remarks | Conclusions and limitations |

### `Real_Data_Validation.ipynb` (companion notebook)

| Section | Content |
|---------|---------|
| Setup | Fetch the real ULB Credit Card Fraud dataset via `sklearn.datasets.fetch_openml` |
| Features | Light feature prep (log-amount; no engineering needed on the pre-anonymized PCA columns) |
| Split | Random 70/30 split on normal transactions only (no timestamp in this dataset) |
| Model | Isolation Forest, same hyperparameters as the main project, contamination = real fraud rate |
| Evaluation | ROC-AUC, PR-AUC, recall, confusion matrix against real fraud labels |
| Conclusion | Comparison against the synthetic project's results |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl
```

---

## Usage

**Main project:**
1. Place `financial_anomaly_data.xlsx` in the same directory as the notebook.
2. Open `Anomaly_detection..ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells in order (`Kernel → Restart & Run All`).

**Real-data validation:**
1. Open `Real_Data_Validation.ipynb` — no manual download needed, it fetches the real fraud
   dataset itself on first run (via `sklearn.datasets.fetch_openml`, cached locally afterward).
2. Run all cells in order.

---

## Author

**Nobukhosi Sibanda** — [GitHub](https://github.com/Nobby-sibanda)
