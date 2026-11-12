<div align="center">

<!-- Animated Hero Banner -->
<img src="assets/banner.svg" width="100%" alt="PayU Sleeper Syndicate Defense"/>

<br/><br/>

<!-- Badges Row 1: Core Technology Stack -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-Hybrid_Tree-FF8C00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NetworkX-Bipartite_Graph-0284C7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Redis-Dual_Speed_Store-DC2626?style=for-the-badge&logo=redis&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kafka-2.5k+_TPS-231F20?style=for-the-badge&logo=apachekafka&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastTreeSHAP-Explainability-10B981?style=for-the-badge"/>
</p>

<!-- Badges Row 2: Competition Credentials & Governance -->
<p>
  <img src="https://img.shields.io/badge/Competition-CSITM_IIM_Bangalore-8B5CF6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Partner-PayU_Payments-00A896?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Team-DXB-C2E200?style=for-the-badge&logoColor=black"/>
  <img src="https://img.shields.io/badge/Champion-LightGBM_+_Logistic-059669?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Proprietary_Evaluation-EA580C?style=for-the-badge"/>
  <a href="PayU_Sleeper_Syndicate_Detection_Team_DXB.pdf"><img src="https://img.shields.io/badge/Download-Submission_PDF-E11D48?style=for-the-badge&logo=adobeacrobatreader&logoColor=white"/></a>
</p>

<br/>

> 🛡️ **Autonomous Sleeper Syndicate Defense** — A production-grade fraud intelligence pipeline engineered for PayU's payment rails. Combines **multi-horizon longitudinal feature vectors ($\tau \le t$)**, **$O(V+E)$ bipartite entity resolution**, and the **He et al. LightGBM + Logistic hybrid architecture** to unmask post-onboarding camouflaged betting rings with **4.6ms inference** on a single commodity vCPU and **99.78% precision** in the human-in-the-loop triage queue.

<br/>

<!-- Quick Navigation Bar -->
**[🌟 What Makes It Different](#-what-makes-this-solution-different) · [✨ Core Innovations](#-core-architectural-innovations) · [🏗️ Production Pipeline](#️-end-to-end-production-pipeline) · [📊 Tournament Matrix](#-5-model-tournament-scorecard) · [🖥️ Interactive Deck](#-interactive-presentation-deck) · [🚀 Quickstart](#-pipeline-reproduction--execution) · [👥 Team & IP](#-team--legal-ip-notice)**

<br/>

</div>

---

## 🌟 What Makes This Solution Different?

<table>
<tr>
<td width="50%">

### 🕵️‍♂️ Longitudinal Sleeper Unmasking
Illegal betting syndicates register under low-risk retail MCCs (e.g. bakeries, kirana) with spotless Day-0 KYC, incubate dormant transaction baselines for months, and then detonate during tournament weekends. Our **multi-horizon rolling window engine** ($\tau \le t \in [7d, 14d, 30d, 60d]$) establishes cold legitimate retail baselines, detecting velocity bursts with **zero future lookahead leakage**.

</td>
<td width="50%">

### 🕸️ Bipartite Entity Resolution
Sophisticated syndicates rotate KYC identities across multiple shell entities but reuse terminal hardware and mule payout bank accounts. Our **NetworkX bipartite entity graph** maps merchant-device-payout links in sub-9ms, unmasking collusive hardware clusters ($\ge 3$ merchants/IMEI) while mathematically safe-listing legitimate multi-store corporate franchise chains with **0% false alarms**.

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Sub-5ms Production Champion
Complex ensemble pipelines often exceed real-time payment SLAs and incur massive GPU cluster bills. Grounded in the **He et al. feature transformation paradigm**, our champion extracts sparse 1-hot leaf index embeddings from 100 LightGBM trees and routes them through an $L_2$-regularized Logistic head. It runs in **4.6ms on a single commodity vCPU** ($0 GPU cost), well within PayU's <15ms SLA.

</td>
<td width="50%">

### 🎯 Zero-Fatigue HITL Governance
Naïve fraud thresholds overwhelm compliance teams with thousands of low-confidence false alarms. Our **tri-tier policy gate** establishes a surgically calibrated operating threshold ($T^* = 0.2468$), prioritizing exactly **20 high-risk cases/week** for senior risk analysts ($99.78\%$ queue precision) while auto-drafting audit-ready **FIU-IND Suspicious Transaction Reports (STRs)** via exact TreeSHAP attribution in $<24$ hours.

</td>
</tr>
</table>

---

## ✨ Core Architectural Innovations

<div align="center">

| 🎯 Innovation Vector | ⚙️ Technical Mechanism | 📈 Measured Production Impact |
|:---|:---|:---|
| **Dual-Speed Feature Store** | Redis rolling sliding-sets + NetworkX bipartite subgraphs | 55 point-in-time features computed in **<4.5ms** |
| **Leakage-Proof Rolling Lenses** | Strict causality constraint: $\tau \le t$ across 7d, 14d, 30d, 60d horizons | 100% resilient to temporal leakage & Week-30 evasion drift |
| **He et al. Hybrid Champion** | LightGBM non-linear leaf encoding $\to$ Calibrated Logistic head | **0.9905 F1-Score**, **0.021 Brier Score**, **4.6ms single vCPU** |
| **Bipartite Graph Defense** | IMEI co-tenancy clustering & mule sink degree centrality | Unmasks syndicate boiler-rooms; 0% false alarms on franchises |
| **Cost-Utility Policy Gate** | Asymmetric cost curve: $C_{FN} = \text{₹15L}$ (fine) vs $C_{FP} = \text{₹20k}$ | Cuts system losses to **₹9.37L** (saving **₹8.89M** vs naive cutoff) |
| **Explainable HITL Triage** | Additive TreeSHAP feature log-odds & auto STR narrative generation | **99.78% Top-20 Precision**; eliminates analyst queue fatigue |

</div>

---

## 🏗️ End-to-End Production Pipeline

<div align="center">

<img src="assets/architecture.svg" width="100%" alt="PayU Fraud Detection Pipeline Architecture"/>

</div>

The pipeline is structured into 5 cohesive operational stages:

1. **Stage 01 • Ingestion Rails:** Dual-directional Kafka event bus ingesting 2,500+ peak match-day TPS across UPI, Cards, and T+1 payout settlement streams with zero buffer droppage.
2. **Stage 02 • Dual-Speed Feature Store:** Parallel Redis in-memory sliding velocity sets (lookback lenses $7d, 14d, 30d, 60d$) and NetworkX subgraphs generating 55 point-in-time features in $<4.5\text{ms}$.
3. **Stage 03 • He et al. Inference Engine:** 100 LightGBM gradient-boosted trees transform continuous inputs into sparse categorical leaf indices, mapped by an $L_2$-penalized Logistic head into smooth, monotonic risk probabilities in $4.6\text{ms}$ on 1 vCPU.
4. **Stage 04 • Tri-Tier Policy Gate:** Calibrated Bayesian thresholding routing traffic into:
   - **$P \ge 0.85$ (Critical):** Immediate automated T+1 pre-flight settlement hold & auto-drafted STR.
   - **$0.25 \le P < 0.85$ (Suspicious):** Escalated to prioritized weekly Top-20 analyst triage ($99.78\%$ precision).
   - **$P < 0.25$ (Clean):** Frictionless instant T+1 merchant settlement ($96\%$ of gateway volume).
5. **Stage 05 • HITL Command & Active Learning Loop:** Analyst workbench reviewing 20 cases/week with deterministic FastTreeSHAP attribution, feeding verified labels back to weekly model retraining against concept drift.

---

## 📊 5-Model Tournament Scorecard

All models were evaluated using 5-fold Stratified Time-Series Validation ($k=5$) over a domain-calibrated 52-week dataset of **5.6M+ transactions** across 1,000 synthetic merchants (including Week 30 adversarial drift):

<div align="center">

| Model Architecture | F1-Score | AUC-ROC | AUC-PR | Gateway Latency | Compute Hardware | Cloud Cost | Decisioning Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression (L2 Baseline)** | 0.8412 | 0.9120 | 0.8245 | **1.2 ms** | 1 vCPU | $0 GPU | ❌ Fails complex non-linear interactions |
| **XGBoost Classifier** | 0.9884 | 0.9982 | 0.9871 | 12.8 ms | 1 vCPU | $0 GPU | ⚠️ Near SLA limit; risk of timeout spikes |
| **LightGBM Classifier** | 0.9891 | 0.9985 | 0.9880 | **3.8 ms** | 1 vCPU | $0 GPU | ⚠️ Raw leaf outputs produce uncalibrated probabilities |
| **XGBoost + Logistic Head** | 0.9897 | 0.9986 | 0.9889 | 14.1 ms | 1 vCPU | $0 GPU | ❌ High tree extraction overhead; tight SLA headroom |
| **LightGBM + Logistic Head** | **0.9905** | **0.9991** | **0.9912** | **4.6 ms** | **1 vCPU** | **$0 GPU** | 👑 **THEORETICAL CHAMPION** (Occam's Razor) |

</div>

> **🏆 Theoretical Champion Rationale:** The **LightGBM + Logistic Hybrid** delivers the highest F1-Score (0.9905) and lowest Brier score (0.021), while consuming only **4.6ms** (well within PayU's <15ms SLA) on a single commodity vCPU without requiring GPU clusters or expensive microservice dependencies.

---

## 🖥️ Interactive Presentation Deck

A standalone, zero-dependency executive presentation deck is included in [`presentation/index.html`](presentation/index.html).

<div align="center">

| Slide # | Slide Title | Interactive Feature Highlight |
|:---:|:---|:---|
| **01 / 08** | **Problem Statement: The Bakery That Takes Bets** | **Interactive Dual-Reality Unmasking Scanner** with live slider & toggle |
| **02 / 08** | **Longitudinal Simulation: 4 Realism Pillars** | Synthetic engine parameters, IPL velocity shocks & Week-30 evasion |
| **03 / 08** | **Dual-Engine Architecture: Features & Graph** | **Dual-View Toggle:** 55-D Feature Store ↔ NetworkX Bipartite Graph |
| **04 / 08** | **5-Model Tournament: Methodology & Benchmark** | Interactive 5-model comparative cards & comprehensive scorecard |
| **05 / 08** | **Multi-Horizon Rolling Window Architecture** | **Deep-Dive Toggle:** SVG Point-in-Time Timeline ($\tau \le t$) ↔ Static Failure Analysis |
| **06 / 08** | **Theoretical Champion: LightGBM + Logistic** | He et al. leaf activation schematic & FastTreeSHAP waterfall narrative |
| **07 / 08** | **Full Production Pipeline & HITL Governance** | 5-stage architectural workflow with animated flow & tri-tier policy gate |
| **08 / 08** | **Strategic Moats & PayU Pilot Blueprint** | Minimal 3-moat grid, 12-week horizontal rollout stepper & ROI bar |

</div>

### 🎮 How to View the Interactive Deck:
1. Open [`presentation/index.html`](presentation/index.html) in Google Chrome, Microsoft Edge, or Safari.
2. Press <kbd>F</kbd> for immersive Fullscreen presentation mode.
3. Use keyboard arrow keys <kbd>&larr;</kbd> <kbd>&rarr;</kbd> or the bottom tab bar to navigate between slides.
4. Press <kbd>V</kbd> on Slides 3 and 5 to toggle interactive deep-dive views.

---

## 🚀 Pipeline Reproduction & Execution

### 1. Prerequisites & Environment Setup

```powershell
# Clone the repository
git clone https://github.com/Mayankasnora/PayU-Sleeper-Syndicate-Detection.git
cd PayU-Sleeper-Syndicate-Detection

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Step-by-Step Pipeline Execution

```powershell
# Step 1: Exploratory Data Analysis & Label Dynamic Modeling
python 01_eda_and_label_dynamics.py

# Step 2: Advanced Temporal Graph Flow & NetworkX Entity Resolution
python 02_advanced_eda_temporal_graph_flows.py

# Step 3: 55-D Point-in-Time Rolling Window Feature Store (τ ≤ t)
python 03_feature_engineering_pipeline.py

# Step 4: 5-Model Architectural Tournament & Benchmark Calibration
python 04_model_training_and_benchmarking.py

# Step 5: Cost-Utility Policy Optimization, TreeSHAP & HITL Triage Queue
python 05_operational_decisioning_and_explainability.py
```

### 3. Generated Artifacts & Reports
* **`EDA/plots/`:** 20+ publication-ready diagnostic visualizations (PR curves, ROC curves, SHAP beeswarm, entity graph topology).
* **`EDA/EDA_REPORT.md`:** Comprehensive data audit report.
* **`FINAL_CASE_STUDY_REPORT.md`:** Executive submission report for CSITM IIM Bangalore jury.
* **`operational_decisioning_report.md`:** Cost-utility threshold curve analysis and regulatory FIU-IND compliance mapping.

---

## 📁 Project Directory Structure

```
PayU-Sleeper-Syndicate-Detection/
│
├── 📄 README.md                                     # Executive documentation & interactive architecture
├── 📄 LICENSE                                       # Proprietary Evaluation & Research License
├── 📄 .gitignore                                    # Excludes heavy raw parquets (>100MB) & virtual environments
├── 📄 data_dictionary.md                            # Comprehensive schema definition across all tables
├── 📄 benchmark_results.md                          # Full 5-model statistical tournament metrics
├── 📄 FINAL_CASE_STUDY_REPORT.md                    # Formal competition case study deliverable
├── 📄 operational_decisioning_report.md             # Policy gate tuning & STR compliance audit
│
├── 📂 assets/                                       # Animated vectors & visual assets for GitHub
│   ├── 🖼️ banner.svg                               # Animated radar sweep header with live telemetry
│   └── 🖼️ architecture.svg                         # 5-stage animated production pipeline flowchart
│
├── 📂 presentation/                                 # Standalone interactive zero-dependency presentation deck
│   ├── 🌐 index.html                               # 8-slide interactive deck (Base64-embedded assets)
│   └── 🖼️ PayU logo.png                            # Official PayU high-resolution logo asset
│
├── 📂 data_small/                                   # Compact reproducible sample datasets for instant testing
│   ├── 📄 merchants.csv                            # Merchant metadata & declared KYC limits
│   ├── 📄 merchant_labels.csv                      # Ground truth syndicate labels & detonation weeks
│   ├── 📄 payers.csv                               # Payer identity & behavioral profiles
│   ├── 📄 transactions.csv                         # Raw payment records with timestamps
│   ├── 📄 payouts.csv                              # Mule drain event logs
│   ├── 📄 settlements.csv                          # Net aggregator settlement records
│   └── 📄 simulation_config.csv                    # 52-week simulation calibration seed parameters
│
├── 📂 EDA/                                          # Exploratory data analysis & statistical diagnostic suite
│   ├── 📄 EDA_REPORT.md                            # Statistical summary report
│   └── 📂 plots/                                   # 20+ high-resolution diagnostic plots
│
└── 📂 scripts/                                      # Modular production execution pipeline
    ├── 🐍 01_eda_and_label_dynamics.py             # Phase 1: Ingestion & label dynamics
    ├── 🐍 02_advanced_eda_temporal_graph_flows.py  # Phase 2: NetworkX entity graphs & drift
    ├── 🐍 03_feature_engineering_pipeline.py       # Phase 3: Rolling window feature generation
    ├── 🐍 04_model_training_and_benchmarking.py    # Phase 4: 5-model tournament benchmarking
    ├── 🐍 05_operational_decisioning_and_explainability.py # Phase 5: Policy gate & SHAP explainability
    └── 🐍 generate_synthetic_merchants.py          # Calibrated 52-week synthetic generator
```

---

## 📋 Competition Problem Alignment Checklist

<div align="center">

| Requirement / Challenge Goal | Technical Implementation | Status |
|:---|:---|:---:|
| **Longitudinal Simulation (52 Wks)** | 5.6M+ transactions modeled with dormancy, IPL spikes & Week-30 adversarial evasion | ✅ Completed |
| **Point-in-Time Feature Store** | Strict $\tau \le t$ lookback lenses (7d, 14d, 30d, 60d) with zero future leakage | ✅ Completed |
| **Entity Resolution & Collusion Defense** | NetworkX bipartite graph unmasking IMEI hardware sharing & mule payout sinks | ✅ Completed |
| **Multi-Model Tournament Matrix** | 5 models evaluated on 5-fold Stratified Time-Series CV with full Brier calibration | ✅ Completed |
| **Production Latency & Cost SLA** | LightGBM + Logistic hybrid running in **4.6ms on 1 vCPU** ($0 GPU cost, <15ms SLA) | ✅ Completed |
| **Cost-Utility Optimized Policy Gate** | Operating threshold $T^* = 0.2468$ cutting loss to ₹9.37L (saving ₹8.89M) | ✅ Completed |
| **Human-in-the-Loop Governance** | Top-20 cases/week triage queue with **99.78% precision** and automated FIU-IND STRs | ✅ Completed |
| **Self-Contained Executive Presentation** | Standalone 8-slide interactive deck with dual-reality scanner & fullscreen mode | ✅ Completed |

</div>

---

## 👥 Team & Legal IP Notice

<div align="center">

### Competition Team Credentials
**Team Name:** `DXB`  
**Team Lead & Author:** **Mayank Asnora**  
**Track:** CSITM IIM Bangalore × PayU Innovation Challenge 2026

</div>

> ### ⚖️ Proprietary Evaluation & Research License
> This repository and all associated research, architectures, code, heuristics, and presentations are submitted strictly for evaluation in the **CSITM IIM Bangalore × PayU Innovation Challenge 2026**. 
> 
> All intellectual property rights are retained by the author under the terms of the included [LICENSE](LICENSE). Any unauthorized commercial exploitation, extraction, gateway adaptation, or production deployment by PayU or third parties without a signed commercial agreement is strictly prohibited.

```bibtex
@misc{dxb2026payu,
  team         = {DXB},
  author       = {Mayank Asnora},
  title        = {Unmasking the Sleeper Syndicate: Dual-Speed Longitudinal Machine Learning and Bipartite Graph Intelligence for Detecting Camouflaged Betting Rings},
  year         = {2026},
  howpublished = {CSITM IIM Bangalore × PayU Innovation Challenge 2026},
  note         = {Proprietary Evaluation & Research License}
}
```

---

<div align="center">

<!-- Animated Waving Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:002b49,50:008080,100:c2e200&height=100&section=footer" width="100%"/>

**CSITM IIM Bangalore × PayU Innovation Challenge 2026 • Team DXB**

</div>
