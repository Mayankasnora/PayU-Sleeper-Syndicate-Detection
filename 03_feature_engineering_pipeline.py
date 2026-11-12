#!/usr/bin/env python3
"""
Phase 2 & 3: Graph Entity Resolution & Time-Aware Rolling Feature Engineering Pipeline.

Components:
  1. Graph Entity Resolution (NetworkX):
     - Bipartite entity graph on merchants and shared identifiers (bank, phone, device, IP, address, beneficiary).
     - Extracts syndicate ring cluster metrics and entity co-sharing degrees.
  2. Time-Aware Weekly Snapshot Rolling Feature Engine:
     - Processes 5.6M transactions into daily aggregates, then generates weekly rolling snapshots (14d/30d windows).
     - Computes multi-domain features: Underwriting Arbitrage, Geo-Dispersion, Ticket/Night Signatures,
       Loss-Chasing Velocity, Acquirer Friction, Flow-of-Funds Settlement Drainage.
  3. Leakage-Free Label Tracking:
     - Differentiates active fraud, time-mature known labels (`label_known_date <= snapshot_date`),
       and Out-of-Time split flag (`is_oot` for Weeks 30-52).

Outputs:
  - `data/merchant_graph_features.parquet`
  - `data/merchant_weekly_features.parquet`
"""
import os
import json
import numpy as np
import pandas as pd
import networkx as nx

DATA_DIR = "data"
OUT_GRAPH_PARQUET = os.path.join(DATA_DIR, "merchant_graph_features.parquet")
OUT_WEEKLY_PARQUET = os.path.join(DATA_DIR, "merchant_weekly_features.parquet")


def build_graph_features(merchants, payouts, labels):
    print("\n--- [Step 1/4] Building Entity Resolution Graph (NetworkX) ---")
    B = nx.Graph()
    m_isfraud = labels.set_index("merchant_id")["is_fraud"].to_dict()

    for _, row in merchants.iterrows():
        mid = row["merchant_id"]
        B.add_node(mid, node_type="merchant")
        B.add_edge(mid, f"BA_{row['registered_bank_account_id']}")
        B.add_edge(mid, f"PH_{row['registered_phone_id']}")
        B.add_edge(mid, f"DEV_{row['onboarding_device_id']}")
        B.add_edge(mid, f"IP_{row['onboarding_ip_block']}")
        B.add_edge(mid, f"AD_{row['registered_address_id']}")

    for _, row in payouts.iterrows():
        mid = row["merchant_id"]
        B.add_edge(mid, f"BEN_ACC_{row['beneficiary_account_id']}")

    print(f"Bipartite Graph built: {B.number_of_nodes():,} nodes, {B.number_of_edges():,} edges.")

    components = list(nx.connected_components(B))
    print(f"Connected components extracted: {len(components):,}")

    graph_records = []
    m_dict = merchants.set_index("merchant_id").to_dict("index")

    for mid in merchants["merchant_id"]:
        m_row = m_dict[mid]
        dev_node = f"DEV_{m_row['onboarding_device_id']}"
        bank_node = f"BA_{m_row['registered_bank_account_id']}"
        phone_node = f"PH_{m_row['registered_phone_id']}"
        ip_node = f"IP_{m_row['onboarding_ip_block']}"
        addr_node = f"AD_{m_row['registered_address_id']}"

        shared_dev = B.degree(dev_node) - 1 if dev_node in B else 0
        shared_bank = B.degree(bank_node) - 1 if bank_node in B else 0
        shared_phone = B.degree(phone_node) - 1 if phone_node in B else 0
        shared_ip = B.degree(ip_node) - 1 if ip_node in B else 0
        shared_addr = B.degree(addr_node) - 1 if addr_node in B else 0

        # Component size (count only merchant nodes in the component)
        comp = nx.node_connected_component(B, mid)
        comp_merchants = [n for n in comp if n.startswith("M") and len(n) == 7]
        cluster_size = len(comp_merchants)

        # Cross-merchant beneficiary sharing
        ben_nodes = [n for n in B.neighbors(mid) if n.startswith("BEN_ACC_")]
        shared_bens = 0
        ben_in_degrees = []
        for b in ben_nodes:
            deg = B.degree(b)
            if deg > 1:
                shared_bens += 1
                ben_in_degrees.append(deg - 1)
        max_ben_in_degree = max(ben_in_degrees) if ben_in_degrees else 0
        shared_mule_sink_score = sum(ben_in_degrees) if ben_in_degrees else 0

        # Syndicate indicator: shared hardware device OR shared mule beneficiaries across merchants
        is_syndicate_cluster = int(shared_dev > 0 or shared_bens > 0)
        # Franchise indicator: shared corporate bank/domain but zero shared device
        is_franchise_cluster = int(shared_bank > 0 and shared_dev == 0 and shared_bens == 0)

        graph_records.append({
            "merchant_id": mid,
            "graph_cluster_size": cluster_size,
            "shared_device_count": shared_dev,
            "shared_bank_count": shared_bank,
            "shared_phone_count": shared_phone,
            "shared_ip_count": shared_ip,
            "shared_address_count": shared_addr,
            "shared_beneficiary_count": shared_bens,
            "max_beneficiary_in_degree": max_ben_in_degree,
            "shared_mule_sink_score": shared_mule_sink_score,
            "is_syndicate_structure": is_syndicate_cluster,
            "is_franchise_structure": is_franchise_cluster
        })

    graph_df = pd.DataFrame(graph_records)
    graph_df.to_parquet(OUT_GRAPH_PARQUET, index=False)
    print(f"Saved graph features to {OUT_GRAPH_PARQUET} ({len(graph_df):,} rows).")
    return graph_df


def build_time_aware_features(merchants, payers, tx, settlements, payouts, labels, graph_df):
    print("\n--- [Step 2/4] Aggregating Transactions to Daily Merchant Granularity ---")
    tx_col = tx[tx.txn_direction == "collection"].copy()
    tx_col["date"] = tx_col["txn_timestamp"].dt.date
    tx_col["hour"] = tx_col["txn_timestamp"].dt.hour
    tx_col["is_night"] = (tx_col["hour"] >= 22) | (tx_col["hour"] <= 3)
    tx_col["is_round_100"] = (tx_col["amount"] % 100 == 0)

    # Geo out of city check
    m_city_map = merchants.set_index("merchant_id")["city"].to_dict()
    tx_col["m_city"] = tx_col["merchant_id"].map(m_city_map)
    tx_col["is_out_of_city"] = (tx_col["payer_city"] != tx_col["m_city"]).astype(int)

    # Loss-chasing burst repeat (<30 mins) and escalating amount
    tx_col_sorted = tx_col.sort_values(["merchant_id", "payer_id", "txn_timestamp"])
    tx_col_sorted["prev_ts"] = tx_col_sorted.groupby(["merchant_id", "payer_id"])["txn_timestamp"].shift(1)
    tx_col_sorted["prev_amount"] = tx_col_sorted.groupby(["merchant_id", "payer_id"])["amount"].shift(1)
    tx_col_sorted["diff_sec"] = (tx_col_sorted["txn_timestamp"] - tx_col_sorted["prev_ts"]).dt.total_seconds()
    tx_col_sorted["is_burst_repeat"] = (tx_col_sorted["diff_sec"] < 1800).astype(int)
    tx_col_sorted["is_escalating_burst"] = ((tx_col_sorted["is_burst_repeat"] == 1) & (tx_col_sorted["amount"] >= tx_col_sorted["prev_amount"])).astype(int)

    # Daily aggregation
    g_day = tx_col_sorted.groupby(["merchant_id", "date"])
    daily_tx = pd.DataFrame({
        "tx_count": g_day.size(),
        "amount": g_day["amount"].sum(),
        "round_count": g_day["is_round_100"].sum(),
        "night_count": g_day["is_night"].sum(),
        "out_of_city_count": g_day["is_out_of_city"].sum(),
        "burst_count": g_day["is_burst_repeat"].sum(),
        "escalating_burst_count": g_day["is_escalating_burst"].sum(),
        "fail_count": g_day["status"].apply(lambda s: (s == "failed").sum()),
        "cb_count": g_day["chargeback_flag"].sum(),
        "unique_payers": g_day["payer_id"].nunique()
    }).reset_index()

    # Daily settlements
    settlements["date"] = settlements["settlement_timestamp"].dt.date
    daily_sett = settlements.groupby(["merchant_id", "date"])["settled_amount"].sum().reset_index()

    # Daily payouts
    payouts["date"] = payouts["payout_timestamp"].dt.date
    daily_po = payouts.groupby(["merchant_id", "date"]).agg(
        payout_amount=("payout_amount", "sum"),
        unique_bens=("beneficiary_id", "nunique")
    ).reset_index()

    print(f"Daily transaction aggregates: {len(daily_tx):,} records.")

    print("\n--- [Step 3/4] Generating Weekly Rolling Snapshots (Weeks 2 to 52) ---")
    start_date = pd.Timestamp(tx["txn_timestamp"].min().date())
    m_onboard_map = merchants.set_index("merchant_id")["onboard_date"].to_dict()
    m_declared_vol_map = merchants.set_index("merchant_id")["declared_monthly_volume"].to_dict()
    m_category_map = merchants.set_index("merchant_id")["declared_category"].to_dict()

    # Pre-index daily metrics for rapid 30-day window slicing
    daily_tx["date"] = pd.to_datetime(daily_tx["date"])
    daily_sett["date"] = pd.to_datetime(daily_sett["date"])
    daily_po["date"] = pd.to_datetime(daily_po["date"])

    snapshots = []
    # 52 weekly snapshots
    for week_num in range(2, 53):
        snap_date = start_date + pd.Timedelta(weeks=week_num)
        win_60d_start = snap_date - pd.Timedelta(days=60)
        win_30d_start = snap_date - pd.Timedelta(days=30)
        win_14d_start = snap_date - pd.Timedelta(days=14)
        win_7d_start = snap_date - pd.Timedelta(days=7)
        prev_14d_start = snap_date - pd.Timedelta(days=28)

        # Multi-horizon temporal slices
        tx_60d = daily_tx[(daily_tx["date"] > win_60d_start) & (daily_tx["date"] <= snap_date)]
        tx_30d = daily_tx[(daily_tx["date"] > win_30d_start) & (daily_tx["date"] <= snap_date)]
        tx_14d = daily_tx[(daily_tx["date"] > win_14d_start) & (daily_tx["date"] <= snap_date)]
        tx_7d = daily_tx[(daily_tx["date"] > win_7d_start) & (daily_tx["date"] <= snap_date)]
        tx_prev_14d = daily_tx[(daily_tx["date"] > prev_14d_start) & (daily_tx["date"] <= win_14d_start)]

        sett_30d = daily_sett[(daily_sett["date"] > win_30d_start) & (daily_sett["date"] <= snap_date)]
        po_30d = daily_po[(daily_po["date"] > win_30d_start) & (daily_po["date"] <= snap_date)]

        # Aggregate per merchant across horizons
        tx_60d_m = tx_60d.groupby("merchant_id")["amount"].sum()
        tx_30d_m = tx_30d.groupby("merchant_id").sum(numeric_only=True)
        tx_14d_m = tx_14d.groupby("merchant_id")["amount"].sum()
        tx_7d_m = tx_7d.groupby("merchant_id")["amount"].sum()
        tx_prev_14d_m = tx_prev_14d.groupby("merchant_id")["amount"].sum()

        # Rolling 30d volatility & daily mean for self-normalized Z-scores
        vol_stats_30d = tx_30d.groupby("merchant_id")["amount"].agg(mean_daily="mean", std_daily="std")

        sett_30d_m = sett_30d.groupby("merchant_id")["settled_amount"].sum()
        po_30d_m = po_30d.groupby("merchant_id").agg({"payout_amount": "sum", "unique_bens": "sum"})

        # Active merchants as of snap_date
        for mid in merchants["merchant_id"]:
            onboard = pd.Timestamp(m_onboard_map[mid])
            if onboard > snap_date:
                continue  # Not onboarded yet

            declared_vol = m_declared_vol_map[mid]
            category = m_category_map[mid]
            days_active = (snap_date - onboard).days

            # Multi-horizon volume metrics
            has_tx = mid in tx_30d_m.index
            vol_7d = tx_7d_m.get(mid, 0.0)
            vol_14d = tx_14d_m.get(mid, 0.0)
            vol_30d = tx_30d_m.loc[mid, "amount"] if has_tx else 0.0
            vol_60d = tx_60d_m.get(mid, 0.0)

            # Warm-up safe historical baseline (expanding window up to 60 days)
            effective_lookback_days = min(60, max(7, days_active))
            expected_daily_vol = vol_60d / effective_lookback_days
            expected_7d_vol = expected_daily_vol * 7.0
            expected_14d_vol = expected_daily_vol * 14.0

            # Dynamic velocity surge ratios
            vol_surge_ratio_7d_to_60d = (vol_7d + 10.0) / (expected_7d_vol + 10.0)
            vol_surge_ratio_14d_to_60d = (vol_14d + 20.0) / (expected_14d_vol + 20.0)

            # Volatility (CV) and Self-Normalized Z-Score
            m_mean = vol_stats_30d.loc[mid, "mean_daily"] if mid in vol_stats_30d.index else 0.0
            m_std = vol_stats_30d.loc[mid, "std_daily"] if mid in vol_stats_30d.index else 0.0
            if pd.isna(m_std):
                m_std = 0.0
            roll_vol_cv_30d = m_std / (m_mean + 1.0)
            daily_avg_7d = vol_7d / 7.0
            vol_zscore_30d = (daily_avg_7d - m_mean) / (m_std + 10.0)

            # 30d operational counts
            cnt_30d = tx_30d_m.loc[mid, "tx_count"] if has_tx else 0
            round_cnt = tx_30d_m.loc[mid, "round_count"] if has_tx else 0
            night_cnt = tx_30d_m.loc[mid, "night_count"] if has_tx else 0
            out_city_cnt = tx_30d_m.loc[mid, "out_of_city_count"] if has_tx else 0
            burst_cnt = tx_30d_m.loc[mid, "burst_count"] if has_tx else 0
            esc_burst_cnt = tx_30d_m.loc[mid, "escalating_burst_count"] if has_tx else 0
            fail_cnt = tx_30d_m.loc[mid, "fail_count"] if has_tx else 0
            cb_cnt = tx_30d_m.loc[mid, "cb_count"] if has_tx else 0
            payers_30d = tx_30d_m.loc[mid, "unique_payers"] if has_tx else 0

            # Acceleration: 14d vs prev 14d
            vol_prev_14d = tx_prev_14d_m.get(mid, 0.0)
            vol_accel = (vol_14d - vol_prev_14d) / max(vol_prev_14d, 1000.0)

            # Settlement & Payouts in 30d
            settled_vol = sett_30d_m.get(mid, 0.0)
            payout_vol = po_30d_m.loc[mid, "payout_amount"] if mid in po_30d_m.index else 0.0
            bens_cnt = po_30d_m.loc[mid, "unique_bens"] if mid in po_30d_m.index else 0

            # Computed ratios
            denom_tx = max(cnt_30d, 1)
            is_offline = category in ["kirana", "restaurant", "stationery"]
            out_city_rate = out_city_cnt / denom_tx

            snapshots.append({
                "merchant_id": mid,
                "snapshot_week": week_num,
                "snapshot_date": snap_date.date(),
                "days_active": days_active,
                # Multi-Horizon Rolling Windows & Adaptive Baselines
                "roll_vol_7d": vol_7d,
                "roll_vol_60d": vol_60d,
                "vol_surge_ratio_7d_to_60d": vol_surge_ratio_7d_to_60d,
                "vol_surge_ratio_14d_to_60d": vol_surge_ratio_14d_to_60d,
                "roll_vol_cv_30d": roll_vol_cv_30d,
                "vol_zscore_30d": vol_zscore_30d,
                # Underwriting Arbitrage
                "declared_monthly_volume": declared_vol,
                "roll_vol_30d": vol_30d,
                "vol_to_declared_ratio_30d": vol_30d / max(declared_vol, 1.0),
                "vol_acceleration_14d": vol_accel,
                "roll_tx_count_30d": cnt_30d,
                "mean_ticket_30d": vol_30d / denom_tx,
                # Ticket & Temporal
                "round_amount_share_30d": round_cnt / denom_tx,
                "night_share_30d": night_cnt / denom_tx,
                # Geo-Dispersion
                "out_of_city_share_30d": out_city_rate,
                "offline_geo_anomaly": int(is_offline and out_city_rate > 0.40),
                # Loss-Chasing & Payer Velocity
                "burst_repeat_rate_30d": burst_cnt / denom_tx,
                "escalating_burst_share_30d": esc_burst_cnt / max(burst_cnt, 1),
                "unique_payers_30d": payers_30d,
                "tx_per_payer_30d": cnt_30d / max(payers_30d, 1),
                # Acquirer Friction & Disputes
                "chargeback_rate_30d": cb_cnt / denom_tx,
                "fail_rate_30d": fail_cnt / denom_tx,
                # Flow of Funds & Settlement Drainage
                "roll_settled_vol_30d": settled_vol,
                "roll_payout_vol_30d": payout_vol,
                "payout_to_settlement_ratio_30d": payout_vol / max(settled_vol, 1.0),
                "unique_beneficiaries_30d": bens_cnt,
                "has_payouts_flag": int(payout_vol > 0),
                "offline_with_payouts_anomaly": int(is_offline and payout_vol > 0),
                # Out of Time flag (Post-drift period starting Week 30)
                "is_oot": int(week_num >= 30)
            })

    snap_df = pd.DataFrame(snapshots)
    print(f"Generated {len(snap_df):,} snapshot observation rows across 51 weeks.")

    # -------------------------------------------------------------------------
    # Merge Static Merchant Attributes, Graph Features & Time-Aware Labels
    # -------------------------------------------------------------------------
    print("\n--- [Step 4/4] Merging Graph Features and Leakage-Free Target Labels ---")
    # Static merchant attributes
    m_static = merchants[[
        "merchant_id", "declared_category", "entity_type", "city",
        "business_size_tier", "business_age_at_onboarding_months",
        "website_present", "website_category_match"
    ]].copy()
    m_static["website_category_match"] = m_static["website_category_match"].fillna(False).astype(int)
    m_static["website_present"] = m_static["website_present"].astype(int)

    # Merge static attributes & graph features
    final_df = snap_df.merge(m_static, on="merchant_id", how="left")
    final_df = final_df.merge(graph_df, on="merchant_id", how="left")

    # Time-aware ground truth labels
    lab_dict = labels.set_index("merchant_id").to_dict("index")
    is_fraud_list = []
    fraud_active_list = []
    label_known_list = []
    archetype_list = []

    for _, row in final_df.iterrows():
        mid = row["merchant_id"]
        s_date = row["snapshot_date"]
        l_info = lab_dict[mid]

        is_fraud = l_info["is_fraud"]
        f_start = l_info["fraud_start_date"]
        l_known = l_info["label_known_date"]

        # Was the merchant actively committing fraud as of snapshot_date?
        active = int(is_fraud and (f_start is not None) and (s_date >= f_start))
        # Was the fraud label formally confirmed & known to risk team as of snapshot_date?
        known = int(is_fraud and (l_known is not None) and (s_date >= l_known))

        is_fraud_list.append(int(is_fraud))
        fraud_active_list.append(active)
        label_known_list.append(known)
        archetype_list.append(l_info["archetype"])

    final_df["is_fraud"] = is_fraud_list
    final_df["is_fraud_active"] = fraud_active_list
    final_df["is_label_known"] = label_known_list
    final_df["archetype"] = archetype_list

    final_df.to_parquet(OUT_WEEKLY_PARQUET, index=False)
    print(f"\n[DONE] Pipeline successfully saved feature matrix to {OUT_WEEKLY_PARQUET}!")
    print(f"Matrix shape: {final_df.shape[0]:,} rows x {final_df.shape[1]} columns.")
    print(f"Pre-Drift Train Observations (Weeks 2-29): {(final_df['is_oot'] == 0).sum():,}")
    print(f"Post-Drift OOT Test Observations (Weeks 30-52): {(final_df['is_oot'] == 1).sum():,}")
    print(f"Active Fraud observations: {final_df['is_fraud_active'].sum():,} ({final_df['is_fraud_active'].mean():.2%})")
    print(f"Formally Known Fraud observations: {final_df['is_label_known'].sum():,} ({final_df['is_label_known'].mean():.2%})")

    return final_df


def main():
    print("Loading raw parquet tables...")
    merchants = pd.read_parquet(os.path.join(DATA_DIR, "merchants.parquet"))
    payers = pd.read_parquet(os.path.join(DATA_DIR, "payers.parquet"))
    tx = pd.read_parquet(os.path.join(DATA_DIR, "transactions.parquet"))
    settlements = pd.read_parquet(os.path.join(DATA_DIR, "settlements.parquet"))
    payouts = pd.read_parquet(os.path.join(DATA_DIR, "payouts.parquet"))
    labels = pd.read_parquet(os.path.join(DATA_DIR, "merchant_labels.parquet"))

    graph_df = build_graph_features(merchants, payouts, labels)
    final_df = build_time_aware_features(merchants, payers, tx, settlements, payouts, labels, graph_df)


if __name__ == "__main__":
    main()

