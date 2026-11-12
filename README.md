# Unmasking the Sleeper Syndicate
### Dual-Speed Longitudinal Machine Learning & Bipartite Graph Intelligence for Detecting Camouflaged Betting Rings
**CSITM IIM Bangalore × PayU Innovation Challenge 2026**

---

### Author & Intellectual Property Notice
* **Author:** Mayank Asnora
* **Institution:** Masters' Union (Undergraduate — Data Science & AI, 2029)
* **Email:** mayank.asnora_ugdsai2029@mastersunion.org
* **License:** [Proprietary Evaluation & Research License](LICENSE) (Strictly evaluation-only. All commercial, deployment, and derivative rights reserved.)

> **Important Legal Notice:**  
> This repository and all associated research, architectures, code, and presentations are submitted strictly for evaluation in the **CSITM IIM Bangalore × PayU Innovation Challenge 2026**. Unauthorized commercial use, extraction, production deployment, or distribution without prior written consent from the author is strictly prohibited.

---

## Executive Summary

Payment aggregators face an acute evasion threat: illegal betting syndicates register under low-risk retail Merchant Category Codes (e.g. bakeries, kirana stores) with clean Day-0 KYC, incubate dormant transaction baselines for months, and then detonate during high-velocity sporting tournaments (e.g. IPL) before draining funds through automated payouts.

This repository implements an end-to-end detection and triage pipeline engineered for PayU's production payment rails:

1. **52-Week Domain-Calibrated Simulation (5.6M+ transactions):** Models realistic Indian payment rails (UPI, Cards, Payouts) with sleeper dormancy, tournament velocity spikes, and Week 30 adversarial evasion.
2. **55-Dimensional Point-in-Time Feature Store:** Multi-horizon lookback windows ($\tau \le t \in [7d, 14d, 30d, 60d]$) with strict zero lookahead leakage.
3. **Bipartite Entity Resolution (NetworkX):** Unmasks hardware co-tenancy (shared IMEI devices) and mule cash-out sinks, while mathematically shielding legitimate corporate franchise chains with 0% false alarms.
4. **Theoretical Champion Architecture (LightGBM + Logistic Hybrid):** Based on the He et al. paradigm—extracts non-linear leaf activations and maps them through an $L_2$-regularized Logistic head to provide calibrated risk probabilities in **4.6ms** on a single vCPU (zero expensive GPU overhead, well within PayU's <15ms gateway SLA).
5. **Human-in-the-Loop Governance:** Calibrated tri-tier policy gate routing the Top-20 weekly cases ($99.78\%$ precision) to senior risk investigators with automated, audit-ready FIU-IND Suspicious Transaction Report (STR) reason codes via TreeSHAP.

---

## Interactive Presentation Deck

A standalone, zero-dependency presentation deck is available in [`presentation/index.html`](presentation/index.html):

* **Slide 01:** Problem Statement & Interactive Dual-Reality Unmasking Scanner
* **Slide 02:** 4 Realism Pillars of Longitudinal Simulation
* **Slide 03:** Engineered Feature Store & Bipartite Graph Intelligence (Dual-View)
* **Slide 04:** 5-Model Architectural Tournament & Comparative Matrix
* **Slide 05:** Multi-Horizon Rolling Window Architecture ($\tau \le t$) & Why Rolling Windows
* **Slide 06:** Final Theoretical Champion (LightGBM + Logistic) & Methodological Reality Check
* **Slide 07:** End-to-End Production Pipeline Flow & Tri-Tier Policy Gate
* **Slide 08:** Strategic Defensive Moats, 12-Week PayU Pilot Blueprint & Executive Impact

*To view the presentation, simply open `presentation/index.html` in any web browser and press <kbd>F</kbd> for fullscreen.*

---

## Pipeline Architecture & Execution Flow

```
01_eda_and_label_dynamics.py                     -> Exploratory data analysis & label definitions
02_advanced_eda_temporal_graph_flows.py          -> Temporal flow analysis & NetworkX graph exploration
03_feature_engineering_pipeline.py               -> 55-D rolling window feature computation (τ ≤ t)
04_model_training_and_benchmarking.py            -> 5-model tournament (Logistic, LightGBM, XGBoost, Hybrids)
05_operational_decisioning_and_explainability.py -> Tri-tier threshold tuning, TreeSHAP, and HITL triage queue
```

---

## Key Performance Indicators

| Metric | Measured Value | Operational Impact |
| :--- | :--- | :--- |
| **Inference Latency** | **4.6 ms** | Well within PayU's <15ms SLA on single commodity vCPU |
| **Top-20 Queue Precision** | **99.78%** | 20 cases/week capacity eliminates analyst alert fatigue |
| **False Positive Rate** | **0.06%** | Protects genuine merchants and corporate franchises |
| **Estimated Loss Avoidance** | **₹8.89M** | Significant reduction in fine exposure vs. naive cutoff |
| **Audit SLA** | **< 24 Hours** | Deterministic TreeSHAP attribution generates plain-English STR filings |

---

## Copyright & Citation

```bibtex
@misc{asnora2026unmasking,
  author = {Asnora, Mayank},
  title  = {Unmasking the Sleeper Syndicate: Dual-Speed Machine Learning and Graph Intelligence for Camouflaged Merchant Detection},
  year   = {2026},
  institution = {Masters' Union},
  note   = {Presented for CSITM IIM Bangalore × PayU Innovation Challenge 2026}
}
```
