# Phase 5: Operational Decisioning, Cost-Utility & Explainability Report

## 1. Operational Executive Summary

This report establishes the production decisioning framework, cost-utility optimization, and automated regulatory explainability engine for the PayU merchant betting fraud detection system.

- **Out-of-Time Test Set:** 21,862 merchant-week snapshot records (Weeks 30 to 52).
- **Target Operational Standard:** Strict False Positive Rate <= 1.0% to safeguard legitimate merchant processing.
- **Queue Efficiency:** Across all 23 test weeks, the weekly Top-20 analyst queue achieved an average precision of **99.78%**, directly eliminating investigative backlog.
- **Automated Explainability:** 100% of escalated cases receive audit-ready TreeSHAP reason codes and synthesized compliance narratives for immediate Suspicious Transaction Report (STR) submission to FIU-IND.

---

## 2. Decision Threshold Calibration & Trade-Off Analysis

In a payment aggregator setting, setting an arbitrary classification cutoff (like 0.50) creates severe operational friction or catastrophic regulatory exposure. We calibrated decision thresholds across four distinct operational policies:

### Policy Comparison Matrix

| Operating Policy             | Threshold | Recall  | Precision | FPR   | True Positives | False Positives | False Negatives | Expected Total Loss |
| ---------------------------- | --------- | ------- | --------- | ----- | -------------- | --------------- | --------------- | ------------------- |
| FPR <= 1.0% Policy Cutoff    | 0.2468    | 100.00% | 98.12%    | 0.06% | 677            | 13              | 0               | INR 937,000         |
| Ultra-Strict FPR <= 0.5%     | 0.2468    | 100.00% | 98.12%    | 0.06% | 677            | 13              | 0               | INR 937,000         |
| Cost-Optimal Min-Loss Cutoff | 0.2468    | 100.00% | 98.12%    | 0.06% | 677            | 13              | 0               | INR 937,000         |
| Standard Baseline (T=0.50)   | 0.5000    | 99.11%  | 98.82%    | 0.04% | 671            | 8               | 6               | INR 9,831,000       |

### Operational Policy Takeaway:
1. **The Policy Cutoff ($T = 0.2468$)** enforces the strict mandate of FPR <= 1.0%. At this threshold, the system achieves **100.00% Recall** while maintaining **98.12% Precision**, generating only 13 false positives across 21,862 merchant evaluations.
2. **Financial Cost-Utility:** Because regulatory fines (PMLA/RBI penalties) and card network sanctions for unauthorized betting exceed INR 15 Lakhs ($C_FN$ = INR 1,500,000), while merchant friction costs are approximately INR 20,000 ($C_FP$ = INR 20,000), the cost-optimal operating threshold is **$T = 0.2468$**. This minimizes total system losses to INR 9.4 Lakhs.

![Decision Threshold Trade-Offs](EDA/plots/decision_threshold_tradeoff.png)
![Cost-Utility Optimization Curve](EDA/plots/cost_utility_optimization_curve.png)

---

## 3. TreeSHAP Global Feature Attribution & Directionality

Using exact TreeSHAP attribution on the tree ensemble, we quantify the precise contribution of each engineered risk dimension:

![SHAP Summary Beeswarm](EDA/plots/shap_summary_beeswarm.png)

### Key Feature Drivers:
1. **`vol_to_declared_ratio_30d` (Underwriting Arbitrage)**: High positive values strongly accelerate risk log-odds. When a small merchant declared at INR 50,000/month processes INR 12,00,000, SHAP assigns an immediate +2.5 to +4.8 log-odds boost.
2. **`shared_device_count` & `shared_mule_sink_score` (Entity Resolution Topology)**: Merchants operating from devices shared with >= 5 other accounts or terminating payouts into concentrated mule sinks receive massive risk penalties.
3. **`out_of_city_share_30d` & `offline_geo_anomaly` (Geo-Entropy)**: High out-of-state transaction shares on localized offline categories (Kirana, local bakeries) separate camouflaged betting fronts from genuine retail stores.
4. **`burst_repeat_rate_30d` & `escalating_burst_share_30d` (Loss-Chasing Velocity)**: Rapid successive payment attempts within 30 minutes with escalating denominations uniquely identify betting deposits.
5. **`payout_to_settlement_ratio_30d` (Mule Evacuation)**: Immediate drainage of cleared T+1 settlements via the Payouts API.

---

## 4. Local SHAP Attribution: Case Study Deconstructions

To demonstrate transparency for merchant appeals and law enforcement inquiries, the engine generates local waterfall decompositions:

![Local SHAP Case Studies](EDA/plots/shap_waterfall_case_studies.png)

- **Case Study A (Sleeper Kirana Front)**: Operates legitimately for weeks before experiencing an explosive volume surge. The waterfall reveals that the risk score jumped primarily due to `vol_to_declared_ratio_30d`, `offline_geo_anomaly`, and `night_share_30d`.
- **Case Study B (Syndicate Mule Node)**: A pass-through merchant fronting for an organized betting ring. Risk score is dominated by `shared_device_count`, `shared_mule_sink_score`, and `payout_to_settlement_ratio_30d`.

---

## 5. Automated Compliance Reason Codes & STR Narrative Engine

For every escalated merchant, the pipeline synthesizes a natural-language regulatory narrative adhering to FIU-IND Suspicious Transaction Report standards.

### Reason Code Taxonomy

| Reason Code | Domain | Operational Trigger Condition | Recommended AML Action |
| :--- | :--- | :--- | :--- |
| `UND_SURGE_01` | Underwriting Arbitrage | 30d volume exceeds declared capacity by >3.0x | Request updated audited financial statements & GST returns |
| `GRAPH_MULE_02` | Entity Resolution | Linked to common mule beneficiary sink or >= 3 shared devices | Immediate freeze of outbound settlement payouts |
| `GEO_ANOMALY_03` | Geographic Entropy | Offline retail category with >60% out-of-station payers | Conduct mystery shopping / physical merchant site visit |
| `BURST_CHASE_04` | Bettor Behavior | Rapid deposit velocity (<30 min repeat) with escalating amounts | Flag payer accounts for illegal gaming deposit patterns |
| `DRAIN_SWEEP_05` | Fund Flow Velocity | >80% of settled funds swept out within 4 hours of T+1 credit | Intercept pending batch payouts to beneficiary accounts |

---

## 6. Operational Alert Triage Queue Export

The active review queue is exported to:
- `data/alert_triage_queue.parquet`
- `data/alert_triage_queue.csv`

### Sample Operational Alert Records (Top Alerts Preview)

| snapshot_week | merchant_name | declared_category | risk_score | priority_tier | primary_risk_driver       | is_fraud_ground_truth |
| ------------- | ------------- | ----------------- | ---------- | ------------- | ------------------------- | --------------------- |
| 30            | M000243       | kirana            | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000289       | kirana            | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000228       | stationery        | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000769       | stationery        | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000373       | stationery        | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000369       | kirana            | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000733       | gaming_recharge   | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000426       | stationery        | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000586       | stationery        | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |
| 30            | M000400       | gaming_recharge   | 1.0        | P1_CRITICAL   | vol_to_declared_ratio_30d | 1                     |

### Actionable Triage Protocol:
- **`P1_CRITICAL` (Score >= 85%)**: Immediate automated settlement payout freeze; file FIU-IND STR within 24 hours.
- **`P2_HIGH` (Score >= 24.7%)**: Escalate to Senior Risk Analyst for secondary KYC and GST audit.
- **`P3_WATCHLIST`**: Retain on enhanced automated monitoring for subsequent 14-day cycle.
