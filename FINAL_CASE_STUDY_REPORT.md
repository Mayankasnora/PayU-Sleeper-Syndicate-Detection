# Executive Case Study Report: Illegal Betting & Merchant Masking Detection

**Target Aggregator:** PayU India / Digital Payment Aggregators  
**Problem Domain:** Detecting Unauthorized Betting & Gambling Syndicates Masked Behind Authentic KYC & Registration Documents  
**Evaluation Protocol:** 52-Week Longitudinal Timeline | Out-of-Time (OOT) Post-Drift Test Set (Weeks 30–52)  
**Champion Architecture:** Hybrid Gradient Boosted Tree Feature Transformer (`LightGBM + Logistics`, Facebook / He et al. Paradigm)

---

## 1. Executive Summary & Problem Formulation

Payment aggregators operate under strict regulatory scrutiny (Reserve Bank of India Guidelines for Payment Aggregators, Prevention of Money Laundering Act / PMLA, and Financial Intelligence Unit / FIU-IND directives). A severe systemic risk facing payment aggregators is **illegal online betting operators camouflaging operations behind authentic registration documents**:
- Organized crime syndicates onboard legitimate-looking corporate or proprietorship entities (e.g. neighborhood grocery stores, stationery shops, or cloud kitchens).
- They submit valid PAN, GSTIN, Aadhaar, and bank account proofs, passing static KYC and document verification checks.
- Once onboarded, they route millions of rupees in unauthorized betting deposits through their payment gateway, often operating in "sleeper" mode for weeks before suddenly ramping up volume during high-stakes sporting events (such as cricket tournaments).
- When law enforcement or card networks detect the activity, the aggregator faces severe financial liabilities: sponsor bank debit freezes, card network excessive fraud penalties ($100K+ per program), regulatory license suspension, and legal proceedings.

### The Technical Solution
We engineered and benchmarked an end-to-end, multi-layered machine learning and graph analytics platform:
1. **Longitudinal Behavioral Modeling:** 5.6M transactions across 1,000 merchants and 100,000 payers over 52 weeks.
2. **Bipartite Entity Graph Resolution:** Bipartite graph over shared bank accounts, phone numbers, onboarding devices, IP blocks, and mule beneficiary sinks.
3. **Multi-Horizon Adaptive Rolling Windows:** Dual-horizon velocity surge ratios ($V_{7d} / V_{60d}$), rolling volume coefficient of variation ($\text{CV}_{30d}$), and merchant self-normalized Z-scores to distinguish smooth organic scaling from abrupt syndicate spikes.
4. **Adversarial Out-of-Time (OOT) Benchmark Suite:** 6 distinct machine learning architectures evaluated strictly on Weeks 30–52 after fraudsters dynamically altered deposit denominations and operating hours.
5. **Operational Decisioning & Explainability Engine:** TreeSHAP attribution generating plain-English compliance reason codes for FIU-IND Suspicious Transaction Reports (STRs) and an operational triage queue achieving **99.78% precision**.

---

## 2. Multi-Layered Risk Feature Architecture

| Risk Dimension | Engineered Feature | Formula / Detection Mechanism | Business Significance |
| :--- | :--- | :--- | :--- |
| **Underwriting Arbitrage** | `vol_to_declared_ratio_30d` | $\frac{\text{Vol}_{30\text{d}}}{\text{Declared Volume}}$ | Catches merchants processing 10x–50x beyond declared limits. |
| **Dynamic Velocity Surge** | `vol_surge_ratio_7d_to_60d` | $\frac{V_{7\text{d}} \times (60/7)}{V_{60\text{d}} + \epsilon}$ | Detects sudden step-function jumps vs smooth organic growth. |
| **Episodic Burstiness** | `roll_vol_cv_30d` | $\frac{\sigma_{\text{daily}, 30\text{d}}}{\mu_{\text{daily}, 30\text{d}} + \epsilon}$ | Captures extreme match-day spike-and-crash volatility ($\text{CV} > 1.8$). |
| **Merchant Volatility** | `vol_zscore_30d` | $\frac{V_{7\text{d}}/7 - \mu_{30\text{d}}}{\sigma_{30\text{d}} + \epsilon}$ | Self-normalized Z-score isolating unexpected injections. |
| **Entity Graph Resolution** | `shared_device_count` | $\text{deg}(DEV) - 1$ | Golden fingerprint: Syndicates share hardware across 12+ merchants. |
| **Mule Sink Concentration** | `shared_mule_sink_score` | $\sum (\text{deg}(BEN) - 1)$ | Multiple merchants sweeping funds into identical beneficiary accounts. |
| **Geo-Entropy Mismatch** | `offline_geo_anomaly` | $\mathbb{I}(\text{Offline} \land \text{OutOfCity} > 40\%)$ | Offline kirana/bakery stores receiving nationwide payments. |
| **Loss-Chasing Velocity** | `burst_repeat_rate_30d` | $\frac{\sum \mathbb{I}(\Delta t < 30\text{m})}{\text{Total Txns}}$ | Bettors repeatedly depositing within 30 min at escalating stakes. |
| **Fund Flow Drainage** | `payout_to_settlement_ratio_30d` | $\frac{\text{Payout Volume}_{30\text{d}}}{\text{Settled Volume}_{30\text{d}}}$ | Rapid evacuation of cleared T+1 settlements via Payouts API. |

---

## 3. Benchmark Modeling Scorecard (Out-of-Time Weeks 30–52)

We evaluated 6 architectures on **21,862 out-of-time merchant-week observations** after the Week 30 adversarial evasion tactics shift:

| Rank | Model Architecture | PR-AUC | ROC-AUC | Recall @ 1% FPR | Precision @ Top-20 | F1-Score | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **LightGBM + Logistics** (Hybrid Champion) | **0.9985** | **1.0000** | **100.00%** | **99.78%** | **0.9905** | < 15 ms |
| 2 | **XGBoost + Logistics** (Hybrid) | 0.9985 | 0.9999 | 100.00% | 100.00% | 0.9429 | < 22 ms |
| 3 | **Random Forest** | 1.0000 | 1.0000 | 100.00% | 100.00% | 0.8855 | < 45 ms |
| 4 | **Logistic Regression** (L2 Baseline) | 0.9989 | 1.0000 | 100.00% | 100.00% | 0.9741 | < 2 ms |
| 5 | **XGBoost** (Standalone GBDT) | 0.9986 | 1.0000 | 100.00% | 100.00% | 0.8652 | < 18 ms |
| 6 | **LightGBM** (Standalone GBDT) | 0.9872 | 0.9996 | 99.85% | 97.61% | 0.8645 | < 8 ms |

### Architectural Insights:
1. **The Facebook (He et al.) Hybrid Paradigm Wins:** Extracting one-hot encoded leaf embeddings from gradient boosted trees and feeding them into regularized logistic regression provides the highest operational F1-score (**0.9905**) and probability calibration.
2. **Resistance to Concept Drift:** Standalone rule-based heuristics collapsed post-Week 30 when operators randomized payment amounts. The machine learning pipeline maintained **100% recall** because graph topology, underwriting ratios, and loss-chasing repeat patterns could not be disguised.

---

## 4. Operational Decisioning & AML Compliance Framework

### Optimal Threshold Calibration
- **Policy Cutoff ($T^* = 0.2468$):** Calibrated to guarantee False Positive Rate $\le 1.0\%$. At this cutoff, the system achieves **100.00% Recall** with only a **0.06% False Positive Rate** (only 13 false positives across 21,862 test evaluations).
- **Financial Cost-Utility:** Minimizes total expected losses to **INR 9.37 Lakhs** (compared to INR 98.31 Lakhs under a standard 0.50 cutoff), preventing over **INR 8.89 Million** in regulatory penalties and merchant friction costs.

### Automated FIU-IND Compliance Reason Code Engine
Every escalated merchant alert is enriched with local TreeSHAP attribution, generating plain-English narratives for regulatory Suspicious Transaction Reports (STRs):

| Reason Code | Category | Operational Trigger | AML Action |
| :--- | :--- | :--- | :--- |
| `UND_SURGE_01` | Underwriting Arbitrage | 30d volume exceeds declared limit by >3.0x | Request updated audited financial statements & GST returns |
| `GRAPH_MULE_02` | Entity Resolution | Linked to common mule sink or $\ge 3$ shared devices | Immediate freeze of outbound settlement payouts |
| `GEO_ANOMALY_03` | Geographic Entropy | Offline retail category with >60% out-of-station payers | Conduct mystery shopping / physical merchant site visit |
| `BURST_CHASE_04` | Bettor Behavior | Rapid deposit velocity (<30 min repeat) with escalating amounts | Flag payer accounts for illegal gaming deposit patterns |
| `DRAIN_SWEEP_05` | Fund Flow Velocity | >80% of settled funds swept out within 4 hours of T+1 credit | Intercept pending batch payouts to beneficiary accounts |

### Active Analyst Triage Queue
The exported triage queue (`data/alert_triage_queue.parquet` and `.csv`) delivers **460 prioritized weekly alerts** across 23 test weeks (exactly 20 per week) with an **overall precision of 99.78%**, directly eliminating analyst queue fatigue.

---

## 5. Artifact Directory & Deliverables Index

- **Pipeline Scripts**:
  - [`generate_synthetic_merchants.py`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/generate_synthetic_merchants.py): 52-week calibrated simulation generator.
  - [`01_eda_and_label_dynamics.py`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/01_eda_and_label_dynamics.py): Baseline EDA and label arrival delay analysis.
  - [`02_advanced_eda_temporal_graph_flows.py`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/02_advanced_eda_temporal_graph_flows.py): Sleeper dynamics, drift, and fund flow analysis.
  - [`03_feature_engineering_pipeline.py`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/03_feature_engineering_pipeline.py): Graph module + multi-horizon rolling windows.
  - [`04_model_training_and_benchmarking.py`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/04_model_training_and_benchmarking.py): 6-model benchmark suite on OOT split.
  - [`05_operational_decisioning_and_explainability.py`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/05_operational_decisioning_and_explainability.py): Threshold optimization, TreeSHAP, and triage queue generator.
- **Reports & Documentation**:
  - [`data_dictionary.md`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/data_dictionary.md): Field-by-field reference and ERD diagram.
  - [`EDA/EDA_REPORT.md`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/EDA/EDA_REPORT.md): Comprehensive exploratory data analysis report.
  - [`benchmark_results.md`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/benchmark_results.md): Full comparative modeling scorecard.
  - [`operational_decisioning_report.md`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/operational_decisioning_report.md): Policy thresholds, cost-utility matrices, and queue metrics.
  - [`log.md`](file:///c:/Users/mayan/OneDrive/Desktop/PayU/log.md): Complete chronological activity and decision audit trail.
- **Exported Datasets**:
  - `data/merchant_weekly_features.parquet` (33,526 rows $\times$ 55 columns)
  - `data/merchant_graph_features.parquet` (1,000 rows)
  - `data/alert_triage_queue.parquet` & `data/alert_triage_queue.csv` (460 prioritized alerts)
- **Diagnostic Plots (in `EDA/plots/`)**:
  - Baseline EDA: `null_values_audit.png`, `numerical_distributions_kde.png`, `numerical_distributions_boxplots.png`, `categorical_distributions.png`, `correlation_matrix_pearson.png`, `correlation_matrix_spearman.png`, `pairwise_interactions.png`.
  - Advanced Domain EDA: `advanced_temporal_sleeper_evolution.png`, `advanced_concept_drift_week30.png`, `advanced_match_day_seasonality.png`, `advanced_entity_graph_topology.png`, `advanced_flow_of_funds_drainage.png`, `advanced_payer_loss_chasing_velocity.png`.
  - Benchmarks & Decisioning: `benchmark_pr_curves.png`, `benchmark_roc_curves.png`, `benchmark_weekly_top20_precision.png`, `benchmark_feature_importance.png`, `decision_threshold_tradeoff.png`, `cost_utility_optimization_curve.png`, `shap_summary_beeswarm.png`, `shap_waterfall_case_studies.png`.

