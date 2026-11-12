#!/usr/bin/env python3
"""
Synthetic payment-platform dataset for illegal-betting merchant detection.

Tables written (parquet if pyarrow is installed, else csv):
  merchants, payers, transactions, settlements, payouts      <- model inputs
  merchant_labels, simulation_config                         <- ground truth, NEVER features

Usage:
  python generate_synthetic_merchants.py --merchants 1000 --weeks 52 --out data
  python generate_synthetic_merchants.py --merchants 300 --weeks 8 --out data_small  # quick test
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    seed: int = 42
    n_merchants: int = 1000
    n_weeks: int = 52
    n_payers: int = 100_000
    start_date: str = "2025-01-06"          # a Monday
    fraud_rate: float = 0.03
    ring_share: float = 0.70                # share of fraud merchants that sit in rings
    sleeper_share: float = 0.30             # share of fraud merchants that are sleepers
    drift_week: int = 30                    # fixed_tier operators change tactics here
    label_rate: float = 0.70                # share of frauds that ever get labeled
    label_delay_weeks: tuple = (4, 8)
    false_label_rate: float = 0.01          # legit merchants wrongly labeled as fraud
    n_franchise_groups: int = 20
    n_mall_groups: int = 20


# --------------------------------------------------------------------------- #
# Static world definition
# --------------------------------------------------------------------------- #
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
          "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Guwahati", "Indore"]
CITY_P = np.array([.16, .15, .13, .09, .08, .08, .07, .06, .05, .05, .04, .04])
CITY_P = CITY_P / CITY_P.sum()

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Rahul", "Amit", "Sanjay", "Vikram",
         "Priya", "Anita", "Neha", "Pooja", "Kavya", "Riya", "Sneha", "Divya", "Meera", "Isha",
         "Manoj", "Suresh", "Rakesh", "Deepak", "Nikhil", "Gaurav", "Ankit", "Harsh", "Yash", "Varun"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Reddy", "Nair", "Iyer", "Das",
        "Bose", "Mehta", "Joshi", "Rao", "Chopra", "Malhotra", "Bhat", "Pillai", "Saha", "Yadav"]
UPI_BANKS = ["okaxis", "oksbi", "ybl", "paytm", "okhdfcbank", "ibl"]
GENERIC_EMAIL = ["gmail.com", "yahoo.in", "outlook.com", "rediffmail.com"]

METHODS = ["upi", "card", "netbanking", "wallet"]
CHANNELS = ["qr", "payment_link", "in_app", "checkout"]
DIRECTIONS = ["collection", "payout"]
STATUSES = ["success", "failed", "reversed"]
FAIL_REASONS = [None, "insufficient_funds", "bank_timeout", "upi_limit", "declined"]
REMARKS = ["", "bill", "order", "thanks", "payment", "invoice", "recharge", "topup",
           "refund", "dep", "bal add", "pts", "wallet", "withdrawal"]
REM = {s: i for i, s in enumerate(REMARKS)}


def hour_profile(peaks, floor=0.02):
    h = np.arange(24)
    p = np.full(24, float(floor))
    for mu, sd, w in peaks:
        d = np.minimum(np.abs(h - mu), 24 - np.abs(h - mu))
        p += w * np.exp(-0.5 * (d / sd) ** 2)
    return p / p.sum()


def vec(d, order):
    v = np.array([d.get(k, 0.0) for k in order], dtype=float)
    return v / v.sum()


BASE_DAILY = {"micro": 6, "small": 20, "medium": 70}
SIZE_TIERS = ["micro", "small", "medium"]
SIZE_P = [0.5, 0.35, 0.15]

PROFILES = {
    "kirana": dict(
        median=150, sigma=0.9, snap=0.5, mean_ticket=200, descriptor="GENERAL STORE", is_offline=True,
        hours=hour_profile([(9, 2, .3), (18, 2.5, .7)]), dow=[1, 1, 1, 1, 1.1, 1.2, 1.0],
        methods=vec(dict(upi=.8, card=.1, wallet=.1), METHODS),
        channels=vec(dict(qr=.85, payment_link=.05, in_app=.05, checkout=.05), CHANNELS),
        zipf=1.2, pool=dict(micro=150, small=500, medium=1500),
        remarks=["", "", "", "bill", "thanks"],
        p_payout=0.0, n_ben=0),
    "restaurant": dict(
        median=450, sigma=0.7, snap=0.5, mean_ticket=500, descriptor="RESTAURANT", is_offline=True,
        hours=hour_profile([(13, 1.5, .4), (20, 2, .4), (23, 2, .2)]), dow=[.9, .9, .9, 1, 1.2, 1.5, 1.4],
        methods=vec(dict(upi=.6, card=.35, wallet=.05), METHODS),
        channels=vec(dict(qr=.6, payment_link=.05, in_app=.3, checkout=.05), CHANNELS),
        zipf=0.9, pool=dict(micro=300, small=900, medium=2500),
        remarks=["", "", "", "bill", "order", "thanks"],
        p_payout=0.0, n_ben=0),
    "stationery": dict(
        median=200, sigma=0.8, snap=0.5, mean_ticket=260, descriptor="STATIONERS", is_offline=True,
        hours=hour_profile([(11, 2, .5), (16, 2, .5)]), dow=[1.2, 1.2, 1.2, 1.2, 1.2, 1.0, .3],
        methods=vec(dict(upi=.75, card=.15, wallet=.1), METHODS),
        channels=vec(dict(qr=.8, payment_link=.05, in_app=.05, checkout=.1), CHANNELS),
        zipf=1.0, pool=dict(micro=120, small=400, medium=1200),
        remarks=["", "", "", "bill", "order", "invoice"],
        p_payout=0.0, n_ben=0),
    "saas": dict(
        tiers=[299, 499, 999, 1999, 4999], tier_p=[.3, .3, .2, .15, .05], mean_ticket=900,
        descriptor="SOFTWARE SERVICES", is_offline=False,
        hours=hour_profile([], 1.0), dow=[1.1, 1.1, 1.1, 1.1, 1.1, .6, .5],
        methods=vec(dict(upi=.3, card=.6, netbanking=.1), METHODS),
        channels=vec(dict(checkout=.9, payment_link=.1), CHANNELS),
        zipf=1.5, pool=dict(micro=60, small=200, medium=700),
        remarks=["", "", "invoice", "payment"],
        p_payout=0.05, k=(1, 2), pay_frac=(.05, .20), n_ben=2),
    "sports_merch": dict(   # cricket-season spikes
        median=800, sigma=0.9, snap=0.5, mean_ticket=1000, descriptor="SPORTS GOODS", is_offline=False,
        hours=hour_profile([(12, 3, .4), (20, 2.5, .6)]), dow=[1, 1, 1, 1, 1.1, 1.3, 1.3],
        match_hours=hour_profile([(13, 3, .4), (20, 2.5, .6)]), match_mult=2.5,
        methods=vec(dict(upi=.7, card=.3), METHODS),
        channels=vec(dict(qr=.4, in_app=.2, checkout=.4), CHANNELS),
        zipf=1.0, pool=dict(micro=100, small=350, medium=1000),
        remarks=["", "", "order", "thanks"],
        p_payout=0.0, n_ben=0),
    "gaming_recharge": dict(   # round amounts, late nights
        tiers=[50, 100, 200, 500, 1000], tier_p=[.3, .3, .2, .15, .05], mean_ticket=230,
        descriptor="GAMING RECHARGE", is_offline=False,
        hours=hour_profile([(22, 2.5, .6), (15, 3, .4)]), dow=[1, 1, 1, 1, 1.1, 1.3, 1.3],
        match_mult=1.8,
        methods=vec(dict(upi=.9, wallet=.1), METHODS),
        channels=vec(dict(in_app=.5, checkout=.4, payment_link=.1), CHANNELS),
        zipf=1.1, pool=dict(micro=200, small=700, medium=2000),
        remarks=["", "recharge", "topup", "recharge"],
        p_payout=0.35, k=(1, 4), pay_frac=(.2, .5), n_ben=5),
    "marketplace": dict(   # many payouts to vendors
        median=600, sigma=1.0, snap=0.5, mean_ticket=800, descriptor="MARKETPLACE", is_offline=False,
        hours=hour_profile([(11, 4, .5), (20, 3, .5)]), dow=[1, 1, 1, 1, 1, 1.1, 1.1],
        methods=vec(dict(upi=.6, card=.4), METHODS),
        channels=vec(dict(checkout=.7, in_app=.3), CHANNELS),
        zipf=1.0, pool=dict(micro=200, small=800, medium=2500),
        remarks=["", "", "order", "payment"],
        p_payout=0.85, k=(3, 8), pay_frac=(.3, .7), n_ben=12),
    "event": dict(   # sudden volume bursts
        median=500, sigma=0.8, snap=0.5, mean_ticket=600, descriptor="EVENTS", is_offline=False,
        hours=hour_profile([(11, 3, .4), (18, 3, .6)]), dow=[1, 1, 1, 1, 1.2, 1.4, 1.4],
        methods=vec(dict(upi=.7, card=.3), METHODS),
        channels=vec(dict(checkout=.5, qr=.3, payment_link=.2), CHANNELS),
        zipf=0.9, pool=dict(micro=200, small=700, medium=2000),
        remarks=["", "", "order", "thanks"],
        p_payout=0.20, k=(1, 3), pay_frac=(.2, .5), n_ben=4),
}
LEGIT_CATS = list(PROFILES)
LEGIT_CAT_P = [.27, .20, .13, .10, .08, .08, .07, .07]
FRAUD_DECLARED_CATS = ["kirana", "stationery", "saas", "restaurant", "gaming_recharge"]
FRAUD_DECLARED_P = [.3, .25, .2, .15, .1]

FRAUD = {
    "fixed_tier": dict(
        tiers=[100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 5000],
        tier_p=[.20, .10, .18, .07, .18, .07, .10, .05, .03, .02],
        hours=hour_profile([(15, 3, .35), (20, 2.5, .45), (1, 2, .2)]),
        hours_post=hour_profile([(14, 3, .4), (19, 3, .4), (23, 2, .2)]),
        match_hours=hour_profile([(16, 2.5, .4), (20, 2, .6)]), match_mult=2.5,
        dow=[.9, .9, .9, .9, 1.1, 1.4, 1.4],
        methods=vec(dict(upi=.9, wallet=.05, netbanking=.05), METHODS),
        channels=vec(dict(payment_link=.5, qr=.3, in_app=.2), CHANNELS),
        zipf=0.8, vol_mult=(2.5, 7.0), payout_frac=0.30,
        p_payout=0.5, k=(1, 4), pay_frac=(.4, .8), delay_h=(1, 24), n_ben=(3, 12)),
    "pass_through": dict(
        median=2500, sigma=1.2,
        hours=hour_profile([(11, 4, .4), (22, 3, .6)]), hours_post=None,
        match_hours=hour_profile([(12, 4, .4), (22, 3, .6)]), match_mult=1.5,
        dow=[1, 1, 1, 1, 1, 1.1, 1.1],
        methods=vec(dict(upi=.7, netbanking=.2, card=.1), METHODS),
        channels=vec(dict(payment_link=.5, qr=.3, in_app=.2), CHANNELS),
        zipf=1.0, vol_mult=(2.0, 6.0), payout_frac=0.10,
        p_payout=0.95, k=(3, 10), pay_frac=(.6, .95), delay_h=(0.1, 4), n_ben=(8, 25)),
}
FRAUD_REMARKS = ["dep", "topup", "bal add", "pts", "wallet", "recharge"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fmt_ids(prefix, arr, width=7):
    arr = np.asarray(arr)
    uniq, inv = np.unique(arr, return_inverse=True)
    strs = np.array([f"{prefix}{int(u):0{width}d}" for u in uniq], dtype=object)
    return strs[inv]


def ip_str(n):
    n = int(n)
    return f"{20 + n % 200}.{(n // 200) % 250}.{(n // 50000) % 250}.0/24"


_zipf_cache = {}


def zipf_w(k, s):
    key = (k, round(s, 3))
    if key not in _zipf_cache:
        w = 1.0 / np.arange(1, k + 1) ** s
        _zipf_cache[key] = w / w.sum()
    return _zipf_cache[key]


def sample_amount(prof, n, rng):
    if "tiers" in prof:
        return rng.choice(prof["tiers"], n, p=prof["tier_p"]).astype(float)
    x = rng.lognormal(np.log(prof["median"]), prof["sigma"], n)
    snap = rng.random(n) < prof.get("snap", 0.5)
    x = np.where(snap, np.round(x / 10) * 10, np.round(x))
    # Natural round ticket billing (e.g., ₹100, ₹200, ₹500 grocery pack/thali)
    snap100 = rng.random(n) < 0.16
    x = np.where(snap100, np.round(x / 100) * 100, x)
    return np.maximum(x, 5.0)


def fraud_amounts(behavior, n, post, noise, rng):
    fp = FRAUD[behavior]
    if behavior == "fixed_tier":
        a = rng.choice(fp["tiers"], n, p=fp["tier_p"]).astype(float)
        jit = rng.random(n) < noise
        a = np.where(jit, np.round(a * (1 + rng.uniform(-.08, .08, n))), a)
        if post.any():   # post-drift: operators randomize amounts
            a[post] = np.round(a[post] * rng.lognormal(0, .35, post.sum()) / 10) * 10
        return np.maximum(a, 10.0)
    x = rng.lognormal(np.log(fp["median"]), fp["sigma"], n)
    snap100 = rng.random(n) < .2
    return np.maximum(np.where(snap100, np.round(x / 100) * 100, np.round(x / 10) * 10), 100.0)


def fill(arr, mask, p, rng):
    k = int(mask.sum())
    if k:
        arr[mask] = rng.choice(24, size=k, p=p)


def pick_codes(probs, n, rng):
    return (rng.random(n)[:, None] > np.cumsum(probs)[None, :-1]).sum(1).astype(np.int8)


# --------------------------------------------------------------------------- #
# World construction
# --------------------------------------------------------------------------- #
class World:
    def __init__(self, cfg: Config, rng):
        self.cfg, self.rng = cfg, rng
        self.n_days = cfg.n_weeks * 7
        self.start = pd.Timestamp(cfg.start_date)
        self.drift_day = cfg.drift_week * 7
        self.cnt = {k: itertools.count(1) for k in
                    ["bank", "phone", "addr", "mdev", "mip", "domain", "pdev", "pip"]}
        self.dev_pool = int(cfg.n_payers * 0.85)
        self.ip_pool = 20_000
        self.extra_dev_n = 60_000
        self.cnt["pdev"] = itertools.count(self.dev_pool + self.extra_dev_n)
        self.cnt["pip"] = itertools.count(self.ip_pool)
        self.build_payers()
        self.build_calendar()

    def build_payers(self):
        cfg, rng = self.cfg, self.rng
        n = cfg.n_payers
        self.p_city = rng.choice(len(CITIES), n, p=CITY_P).astype(np.int8)
        fi, li = rng.integers(0, len(FIRST), n), rng.integers(0, len(LAST), n)
        bi = rng.integers(0, len(UPI_BANKS), n)
        self.p_name = np.array([f"{FIRST[a]} {LAST[b]}" for a, b in zip(fi, li)], dtype=object)
        self.p_upi = np.array([f"{FIRST[a].lower()}.{i}@{UPI_BANKS[c]}"
                               for i, (a, c) in enumerate(zip(fi, bi))], dtype=object)
        self.p_card = np.array([f"CT{i:08d}" for i in range(n)], dtype=object)
        self.p_bank = np.array([f"PBA{i:09d}" for i in range(n)], dtype=object)
        self.p_wallet = np.array([f"WL{i:08d}" for i in range(n)], dtype=object)
        self.p_dev = rng.integers(0, self.dev_pool, n)
        self.p_ip = rng.integers(0, self.ip_pool, n)
        self.p_first_seen = self.start - pd.to_timedelta(rng.integers(0, 720, n), unit="D")
        self.by_city = {c: np.flatnonzero(self.p_city == c) for c in range(len(CITIES))}

    def build_calendar(self):
        rng = self.rng
        self.match = np.zeros(self.n_days, dtype=bool)
        for lo, hi in [(8, 14), (34, 42)]:       # two "cricket seasons"
            for w in range(lo, min(hi, self.cfg.n_weeks)):
                self.match[w * 7 + rng.choice(7, 4, replace=False)] = True
        # Intermittent bank/acquirer network glitch days affecting the entire ecosystem
        self.bank_glitch = rng.random(self.n_days) < 0.05

    def pool(self, city, k, city_share=0.85):
        rng = self.rng
        k = max(int(k), 5)
        local = self.by_city[city]
        n_local = min(int(k * city_share), len(local))
        a = rng.choice(local, n_local, replace=False)
        b = rng.choice(self.cfg.n_payers, k - n_local, replace=False)
        return np.unique(np.concatenate([a, b]))


# --------------------------------------------------------------------------- #
# Merchant specs
# --------------------------------------------------------------------------- #
def build_merchants(w: World):
    cfg, rng = w.cfg, w.rng
    n = cfg.n_merchants
    n_fraud = int(round(n * cfg.fraud_rate))
    perm = rng.permutation(n)
    fraud_ids = set(perm[:n_fraud].tolist())
    legit_ids = perm[n_fraud:]

    # ---- rings
    ring_members = int(n_fraud * cfg.ring_share)
    fraud_list = perm[:n_fraud].tolist()
    rings, ring_of, pos = [], {}, 0
    while pos < ring_members:
        size = int(min(rng.integers(5, 26), ring_members - pos))
        if size < 3:
            size = ring_members - pos
        r0 = int(rng.integers(1, max(2, int(w.n_days * 0.4))))
        city = int(rng.integers(0, len(CITIES)))
        ring = dict(
            id=f"R{len(rings) + 1:03d}", start=r0, city=city,
            bank=next(w.cnt["bank"]), phone=next(w.cnt["phone"]), addr=next(w.cnt["addr"]),
            mdev=next(w.cnt["mdev"]), mip=next(w.cnt["mip"]),
            domain=f"ent{len(rings) + 1}.in",
            ben_pool=np.array([next(w.cnt["bank"]) for _ in range(int(rng.integers(3, 7)))]),
            payer_pool=rng.choice(cfg.n_payers, int(rng.integers(1500, 5000)), replace=False),
            dev_pool=np.array([next(w.cnt["pdev"]) for _ in range(int(rng.integers(20, 80)))]),
            ip_pool=np.array([next(w.cnt["pip"]) for _ in range(int(rng.integers(10, 40)))]),
        )
        for m in fraud_list[pos:pos + size]:
            ring_of[m] = ring
        rings.append(ring)
        pos += size

    # ---- hard-negative legit groups
    legit_list = legit_ids.tolist()
    rng.shuffle(legit_list)
    franchise_of, mall_of, cur = {}, {}, 0
    for g in range(cfg.n_franchise_groups):
        size = int(rng.integers(3, 9))
        grp = dict(bank=next(w.cnt["bank"]), phone=next(w.cnt["phone"]), domain=f"brand{g + 1}.in")
        for m in legit_list[cur:cur + size]:
            franchise_of[m] = grp
        cur += size
    for g in range(cfg.n_mall_groups):
        size = int(rng.integers(4, 11))
        city = int(rng.integers(0, len(CITIES)))
        grp = dict(ip=next(w.cnt["mip"]), addr=next(w.cnt["addr"]), city=city,
                   payers=w.pool(city, 600, city_share=0.85))
        for m in legit_list[cur:cur + size]:
            mall_of[m] = grp
        cur += size

    merchants = []
    for i in range(n):
        is_fraud = i in fraud_ids
        ring = ring_of.get(i)
        fr, ml = franchise_of.get(i), mall_of.get(i)
        m = dict(idx=i, merchant_id=f"M{i + 1:06d}", is_fraud=is_fraud, ring=ring)

        if is_fraud:
            m["category"] = str(rng.choice(FRAUD_DECLARED_CATS, p=FRAUD_DECLARED_P))
            # Underwriting arbitrage: fraudsters deliberately declare small/micro to breeze past KYC
            m["size_tier"] = str(rng.choice(["micro", "small", "medium"], p=[.7, .25, .05]))
            m["city"] = (ring["city"] if ring is not None and rng.random() < .7
                         else int(rng.choice(len(CITIES), p=CITY_P)))
        else:
            m["category"] = str(rng.choice(LEGIT_CATS, p=LEGIT_CAT_P))
            m["size_tier"] = str(rng.choice(SIZE_TIERS, p=SIZE_P))
            m["city"] = ml["city"] if ml else int(rng.choice(len(CITIES), p=CITY_P))
        prof = PROFILES[m["category"]]

        max_onboard = max(1, int(w.n_days * 0.85))
        if ring is not None:
            m["onboard"] = int(min(ring["start"] + rng.integers(0, 19), max_onboard))
        elif is_fraud:
            m["onboard"] = int(rng.integers(1, max(2, int(w.n_days * 0.6))))
        else:
            m["onboard"] = int(rng.integers(0, max(1, int(w.n_days * 0.75))))
        m["archetype"], m["behavior"], m["fraud_start"] = "none", None, 10 ** 9
        if is_fraud:
            m["behavior"] = str(rng.choice(["fixed_tier", "pass_through"], p=[.6, .4]))
            sleeper = rng.random() < cfg.sleeper_share
            start = m["onboard"] + int(rng.integers(0, 8))
            if sleeper:
                s2 = m["onboard"] + 7 * int(rng.integers(2, max(3, int(w.n_days / 14))))
                if s2 < w.n_days - 7:
                    start, m["archetype"] = s2, "sleeper"
            if m["archetype"] != "sleeper":
                m["archetype"] = m["behavior"]
            m["fraud_start"] = int(max(m["onboard"], min(start, max(m["onboard"], w.n_days - 7))))
            m["decoy_share"] = float(rng.uniform(0, .6))
            m["amount_noise"] = float(rng.uniform(0, .3))
            lo, hi = FRAUD[m["behavior"]]["vol_mult"]
            m["vol_mult"] = float(rng.uniform(lo, hi))
            # Heterogeneous dispute and risk throttling profiles across fraud operators
            m["dispute_propensity"] = float(rng.choice([0.0012, 0.004, 0.015], p=[0.50, 0.35, 0.15]))
            m["risk_scrutiny"] = float(rng.choice([0.0, 0.04, 0.18], p=[0.65, 0.23, 0.12]))
        else:
            p_cb = 0.0003 if prof.get("is_offline", False) else 0.002
            m["dispute_propensity"] = float(p_cb)
            m["risk_scrutiny"] = 0.0

        m["pay_scale"] = float(rng.uniform(.3, 1.0)) if is_fraud else 1.0
        m["legit_pay_mult"] = float(np.clip(rng.lognormal(0, .8), .3, 4))

        # Identifiers
        use_ring = lambda p: ring is not None and rng.random() < p
        m["bank"] = ring["bank"] if use_ring(.5) else (fr["bank"] if fr else next(w.cnt["bank"]))
        m["phone"] = ring["phone"] if use_ring(.3) else (fr["phone"] if fr and rng.random() < .5
                                                         else next(w.cnt["phone"]))
        m["addr"] = ring["addr"] if use_ring(.3) else (ml["addr"] if ml else next(w.cnt["addr"]))
        m["mdev"] = ring["mdev"] if use_ring(.7) else next(w.cnt["mdev"])
        m["mip"] = ring["mip"] if use_ring(.8) else (ml["ip"] if ml else next(w.cnt["mip"]))
        etype = str(rng.choice(["sole_proprietor", "partnership", "pvt_ltd"], p=[.55, .25, .2]))
        m["entity_type"] = etype
        if use_ring(.5):
            m["domain"] = ring["domain"]
        elif fr:
            m["domain"] = fr["domain"]
        elif etype == "pvt_ltd":
            m["domain"] = f"co{next(w.cnt['domain'])}.in"
        else:
            m["domain"] = str(rng.choice(GENERIC_EMAIL, p=[.6, .2, .12, .08]))

        m["age_months"] = int(np.clip(rng.lognormal(np.log(12 if is_fraud else 36), 1.0), 1, 240))
        m["declared_volume"] = float(BASE_DAILY[m["size_tier"]] * 30 * prof["mean_ticket"]
                                     * rng.lognormal(0, .3))
        site_p = 1.0 if m["category"] in ("saas", "marketplace") else (.45 if is_fraud else .75)
        m["website_present"] = bool(rng.random() < site_p)
        match_p = .6 if is_fraud else .95
        m["website_match"] = bool(rng.random() < match_p) if m["website_present"] else None
        m["generic_descriptor"] = bool(is_fraud and rng.random() < .3)

        # Geographic dispersion: legit offline stores have high city concentration (0.88),
        # online stores have national spread (0.15).
        # Fraud offline stores also end up with national spread (0.18).
        city_share = 0.88 if prof.get("is_offline", False) else 0.15
        k = prof["pool"][m["size_tier"]] * rng.uniform(.7, 1.3)
        own = w.pool(m["city"], k, city_share=city_share)
        if ml:
            own = np.unique(np.concatenate([own[: len(own) // 2], ml["payers"][: len(ml["payers"]) // 2]]))
        m["pool_legit"] = own
        if is_fraud:
            if ring is not None:
                share = rng.uniform(.2, .5)
                sub = rng.choice(ring["payer_pool"], int(share * len(ring["payer_pool"])), replace=False)
                extra = w.pool(m["city"], rng.integers(300, 1500), city_share=.18)
                m["pool_fraud"] = np.unique(np.concatenate([sub, extra]))
            else:
                m["pool_fraud"] = w.pool(m["city"], rng.integers(800, 4000), city_share=.18)

        # Beneficiaries
        if is_fraud:
            n_ben = int(rng.integers(*FRAUD[m["behavior"]]["n_ben"]))
            base = [m["bank"]] + [next(w.cnt["bank"]) for _ in range(n_ben)]
            if ring is not None:
                base += ring["ben_pool"].tolist()
        else:
            n_extra = prof.get("n_ben", 0)
            base = [m["bank"]] + ([next(w.cnt["bank"]) for _ in range(n_extra)] if n_extra > 0 else [])
        m["ben_pool"] = np.array(base)
        merchants.append(m)
    return merchants, rings


# --------------------------------------------------------------------------- #
# Per-merchant simulation
# --------------------------------------------------------------------------- #
def simulate_merchant(m, w: World, rng):
    n_days = w.n_days
    prof = PROFILES[m["category"]]
    is_fraud = m["is_fraud"]
    ring = m["ring"]
    days = np.arange(m["onboard"], n_days)
    if len(days) == 0:
        return None
    dow = days % 7
    ramp = np.minimum(1.0, 0.3 + 0.7 * (days - m["onboard"]) / 14)
    season = 1 + 0.1 * np.sin(2 * np.pi * days / 365)
    is_match = w.match[days]
    base = BASE_DAILY[m["size_tier"]]

    rate = base * np.array(prof["dow"])[dow] * ramp * season
    if prof.get("match_mult"):
        rate = rate * np.where(is_match, prof["match_mult"], 1.0)
    if m["category"] == "event":
        ev = np.full(len(days), 0.3)
        for _ in range(3):
            s = int(rng.integers(0, len(days)))
            ev[s:s + 3] = 8.0
        rate = rate * ev

    fa = np.zeros(len(days), dtype=bool)
    if is_fraud:
        fp = FRAUD[m["behavior"]]
        fa = days >= m["fraud_start"]
        fr = (base * m["vol_mult"] * np.array(fp["dow"])[dow] * ramp * season
              * np.where(is_match, fp["match_mult"], 1.0))
        rate = np.where(fa, fr, rate)

    counts = rng.poisson(rate)
    N = int(counts.sum())
    if N == 0:
        return None
    dd = np.repeat(days, counts)
    match_t = w.match[dd]
    post_t = dd >= w.drift_day
    ftx = np.zeros(N, dtype=bool)
    if is_fraud:
        ftx = np.repeat(fa, counts) & (rng.random(N) >= m["decoy_share"])

    # ---- hours
    hours = np.empty(N, dtype=np.int64)
    lm = ~ftx
    fill(hours, lm & ~match_t, prof["hours"], rng)
    fill(hours, lm & match_t, prof.get("match_hours", prof["hours"]), rng)
    if ftx.any():
        fp = FRAUD[m["behavior"]]
        hp_post = fp["hours_post"] if fp["hours_post"] is not None else fp["hours"]
        fill(hours, ftx & ~match_t & ~post_t, fp["hours"], rng)
        fill(hours, ftx & ~match_t & post_t, hp_post, rng)
        fill(hours, ftx & match_t, fp["match_hours"], rng)

    sec_jitter = rng.integers(0, 3600, N)
    ts = dd * 86400 + hours * 3600 + sec_jitter

    # ---- amounts
    amount = np.empty(N)
    if lm.any():
        amount[lm] = sample_amount(prof, int(lm.sum()), rng)
    if ftx.any():
        amount[ftx] = fraud_amounts(m["behavior"], int(ftx.sum()), post_t[ftx], m["amount_noise"], rng)

    # ---- payers & "chasing losses" repeat velocity
    payer = np.empty(N, dtype=np.int64)
    pl = m["pool_legit"]
    if lm.any():
        payer[lm] = pl[rng.choice(len(pl), int(lm.sum()), p=zipf_w(len(pl), prof["zipf"]))]
    if ftx.any():
        pf = m["pool_fraud"]
        sampled_payers = pf[rng.choice(len(pf), int(ftx.sum()), p=zipf_w(len(pf), FRAUD[m["behavior"]]["zipf"]))]
        # Repeat deposit burst: ~18% of fraud txns repeat a recently active bettor within 5-25 min
        repeat_mask = rng.random(len(sampled_payers)) < 0.18
        if repeat_mask.any() and len(sampled_payers) > 1:
            prev_idx = np.maximum(0, np.arange(len(sampled_payers)) - rng.integers(1, 4, len(sampled_payers)))
            sampled_payers[repeat_mask] = sampled_payers[prev_idx[repeat_mask]]
        payer[ftx] = sampled_payers

    # ---- method / channel
    method = np.empty(N, dtype=np.int8)
    channel = np.empty(N, dtype=np.int8)
    if lm.any():
        method[lm] = pick_codes(prof["methods"], int(lm.sum()), rng)
        channel[lm] = pick_codes(prof["channels"], int(lm.sum()), rng)
    if ftx.any():
        fp = FRAUD[m["behavior"]]
        method[ftx] = pick_codes(fp["methods"], int(ftx.sum()), rng)
        channel[ftx] = pick_codes(fp["channels"], int(ftx.sum()), rng)

    # ---- status, failure reasons, refunds, chargebacks
    # Ecosystem-wide bank/acquirer hiccups hit all merchants intermittently
    is_glitch = w.bank_glitch[dd]
    base_fail = np.where(is_glitch, 0.075, 0.035)

    # Risk throttling: only applies to merchants under active risk scrutiny
    scrutiny = m.get("risk_scrutiny", 0.0)
    risk_spike = ftx & (rng.random(N) < scrutiny)
    pfail = np.where(risk_spike, 0.28, np.where(ftx, base_fail + 0.012, base_fail))
    u = rng.random(N)
    status = np.where(u < pfail, 1, np.where(u < pfail + .005, 2, 0)).astype(np.int8)

    fail_codes = rng.integers(1, 5, N)
    fail_codes[risk_spike] = rng.choice([3, 4], size=int(risk_spike.sum()))  # upi_limit, declined
    fail = np.where(status == 1, fail_codes, 0).astype(np.int8)

    # Calibrated disputes & refunds
    is_offline = prof.get("is_offline", False)
    p_refund_legit = 0.002 if is_offline else 0.025
    refund = rng.random(N) < np.where(ftx, 0.004, p_refund_legit)

    # Disputes vary realistically per merchant (some fraud merchants have low disputes; others high)
    cb_p = np.where(ftx, m.get("dispute_propensity", 0.004), 0.0003 if is_offline else 0.002)
    chargeback = rng.random(N) < cb_p

    # ---- device / ip / city
    dev = w.p_dev[payer].copy()
    ip = w.p_ip[payer].copy()
    newdev = rng.random(N) < .08
    dev[newdev] = w.dev_pool + rng.integers(0, w.extra_dev_n, int(newdev.sum()))
    if ring is not None and ftx.any():
        r = ftx & (rng.random(N) < .15)
        dev[r] = rng.choice(ring["dev_pool"], int(r.sum()))
        ip[r] = rng.choice(ring["ip_pool"], int(r.sum()))
    city = w.p_city[payer].copy()
    trav = rng.random(N) < .05
    city[trav] = rng.integers(0, len(CITIES), int(trav.sum()))

    # ---- remarks
    leg_rem = np.array([REM[s] for s in prof["remarks"]])
    remark = leg_rem[rng.integers(0, len(leg_rem), N)]
    if ftx.any():
        fr_rem = np.array([REM[s] for s in FRAUD_REMARKS])
        use = ftx & (rng.random(N) < .25)
        remark[use] = fr_rem[rng.integers(0, len(fr_rem), int(use.sum()))]

    cols = dict(ts=ts, payer=payer, amount=amount, method=method, direction=np.zeros(N, np.int8),
                status=status, fail=fail, refund=refund, chargeback=chargeback, channel=channel,
                dev=dev, ip=ip, city=city, remark=remark)

    # ---- payouts to payers (refunds for legit, "winnings" for fraud)
    p_row = np.where(ftx, FRAUD[m["behavior"]]["payout_frac"] if is_fraud else 0,
                     .03 if m["category"] == "marketplace" else .005)
    sel = np.flatnonzero((status == 0) & (rng.random(N) < p_row))
    if len(sel):
        po = {k: v[sel].copy() for k, v in cols.items()}
        po["ts"] = po["ts"] + rng.integers(60, 6 * 3600, len(sel))
        fsel = ftx[sel]
        po["amount"] = np.where(fsel, np.round(po["amount"] * np.clip(rng.lognormal(.3, .5, len(sel)), .3, 6) / 10) * 10,
                                po["amount"])
        po["direction"] = np.ones(len(sel), np.int8)
        po["status"] = np.zeros(len(sel), np.int8)
        po["fail"] = np.zeros(len(sel), np.int8)
        po["refund"] = ~fsel
        po["chargeback"] = np.zeros(len(sel), bool)
        po["remark"] = np.where(fsel, REM["withdrawal"], REM["refund"])
        cols = {k: np.concatenate([cols[k], po[k]]) for k in cols}
    cols["merchant"] = np.full(len(cols["ts"]), m["idx"], dtype=np.int32)

    # ---- settlements (gross successful collections, next day 06:00)
    ok = (status == 0)
    settled = np.bincount((ts[ok] // 86400), weights=amount[ok], minlength=n_days)
    sd = np.flatnonzero(settled > 0)
    s_amt = settled[sd]
    s_ts = (sd + 1) * 86400 + 6 * 3600 + rng.integers(0, 1800, len(sd))
    settle = dict(merchant=np.full(len(sd), m["idx"], np.int32), ts=s_ts, amount=s_amt)

    # ---- payouts to beneficiaries (disbursements via gateway)
    L = len(sd)
    mult = m["legit_pay_mult"]
    legit_p = prof.get("p_payout", 0.0) * mult
    legit_k = prof.get("k", (1, 2))
    lf = prof.get("pay_frac", (.3, .7))
    legit_f = (lf[0], min(.95, lf[1] * max(1.0, mult)))
    fa_d = (sd >= m["fraud_start"]) if is_fraud else np.zeros(L, bool)

    if is_fraud:
        fp = FRAUD[m["behavior"]]
        if is_offline:
            # Offline fraud fronts move funds via bank sweeps, rarely gateway payouts
            p_pay = np.where(fa_d, 0.10 * m["pay_scale"], 0.0)
        else:
            p_pay = np.where(fa_d, fp["p_payout"] * m["pay_scale"], legit_p)
        frac = np.where(fa_d, rng.uniform(*fp["pay_frac"], L) * m["pay_scale"], rng.uniform(*legit_f, L))
        k = np.where(fa_d, rng.integers(fp["k"][0], fp["k"][1] + 1, L),
                     rng.integers(legit_k[0], legit_k[1] + 1, L))
        delay = np.where(fa_d, rng.uniform(*fp["delay_h"], L), rng.uniform(4, 72, L))
    else:
        p_pay = np.full(L, legit_p)
        frac = rng.uniform(*legit_f, L)
        k = rng.integers(legit_k[0], legit_k[1] + 1, L)
        delay = rng.uniform(4, 72, L)

    has = np.flatnonzero(rng.random(L) < p_pay)
    pay = None
    if len(has) and len(m["ben_pool"]) > 0:
        kk = k[has]
        rows = np.repeat(has, kk)
        wts = rng.random(len(rows)) + .1
        starts = np.r_[0, np.cumsum(kk)[:-1]]
        gsum = np.add.reduceat(wts, starts)
        amt = (s_amt * frac)[rows] * wts / np.repeat(gsum, kk)
        pts = s_ts[rows] + (delay[rows] * 3600).astype(np.int64) + rng.integers(0, 1800, len(rows))
        pool = m["ben_pool"]
        lim_legit = min(len(pool), prof.get("n_ben", 1) + 1)
        lim = max(1, int(np.where(fa_d[rows], len(pool), lim_legit).max()))
        ben = pool[(rng.random(len(rows)) * lim).astype(int) % len(pool)]
        pay = dict(merchant=np.full(len(rows), m["idx"], np.int32), ts=pts, amount=amt, ben=ben)
    return cols, settle, pay


# --------------------------------------------------------------------------- #
# Assemble tables
# --------------------------------------------------------------------------- #
def cat(items, key):
    parts = [it[key] for it in items if it is not None]
    return np.concatenate(parts) if parts else np.array([])


def to_ts(sec, w):
    return (w.start + pd.to_timedelta(sec, unit="s")).tz_localize("Asia/Kolkata")


def build_tables(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    w = World(cfg, rng)
    merchants, rings = build_merchants(w)

    tx_parts, st_parts, po_parts = [], [], []
    for m in merchants:
        out = simulate_merchant(m, w, rng)
        if out is None:
            continue
        tx_parts.append(out[0])
        st_parts.append(out[1])
        if out[2] is not None:
            po_parts.append(out[2])

    # ---- transactions
    T = {k: cat([{"v": p[k]} for p in tx_parts], "v") for k in tx_parts[0]}
    order = np.argsort(T["ts"], kind="stable")
    T = {k: v[order] for k, v in T.items()}
    n = len(T["ts"])
    inst = np.where(T["method"] == 0, w.p_upi[T["payer"]],
           np.where(T["method"] == 1, w.p_card[T["payer"]],
           np.where(T["method"] == 2, w.p_bank[T["payer"]], w.p_wallet[T["payer"]])))
    mids = np.array([m["merchant_id"] for m in merchants], dtype=object)
    desc_tbl = np.array([
        (f"GLOBAL SERVICES {m['idx'] + 1:04d}" if m["generic_descriptor"]
         else f"{PROFILES[m['category']]['descriptor']} {CITIES[m['city']][:3].upper()}")
        for m in merchants], dtype=object)
    transactions = pd.DataFrame({
        "txn_id": fmt_ids("TXN", np.arange(1, n + 1), 10),
        "txn_timestamp": to_ts(T["ts"], w),
        "merchant_id": mids[T["merchant"]],
        "payer_id": fmt_ids("P", T["payer"]),
        "payer_instrument_id": inst,
        "amount": T["amount"].round(2),
        "payment_method": pd.Categorical.from_codes(T["method"], METHODS),
        "txn_direction": pd.Categorical.from_codes(T["direction"], DIRECTIONS),
        "status": pd.Categorical.from_codes(T["status"], STATUSES),
        "failure_reason": np.array(FAIL_REASONS, dtype=object)[T["fail"]],
        "refund_flag": T["refund"],
        "chargeback_flag": T["chargeback"],
        "channel": pd.Categorical.from_codes(T["channel"], CHANNELS),
        "merchant_descriptor": desc_tbl[T["merchant"]],
        "payer_remark": np.array(REMARKS, dtype=object)[T["remark"]],
        "payer_device_id": fmt_ids("PD", T["dev"]),
        "payer_ip_block": np.array([ip_str(x) for x in np.unique(T["ip"])], dtype=object)[
            np.unique(T["ip"], return_inverse=True)[1]],
        "payer_city": np.array(CITIES, dtype=object)[T["city"]],
    })

    # ---- merchants
    merchants_df = pd.DataFrame({
        "merchant_id": mids,
        "declared_category": [m["category"] for m in merchants],
        "entity_type": [m["entity_type"] for m in merchants],
        "onboard_date": (w.start + pd.to_timedelta([m["onboard"] for m in merchants], unit="D")).date,
        "city": [CITIES[m["city"]] for m in merchants],
        "business_size_tier": [m["size_tier"] for m in merchants],
        "business_age_at_onboarding_months": [m["age_months"] for m in merchants],
        "declared_monthly_volume": [round(m["declared_volume"], 2) for m in merchants],
        "registered_bank_account_id": fmt_ids("BA", [m["bank"] for m in merchants]),
        "registered_phone_id": fmt_ids("PH", [m["phone"] for m in merchants]),
        "registered_email_domain": [m["domain"] for m in merchants],
        "registered_address_id": fmt_ids("AD", [m["addr"] for m in merchants]),
        "onboarding_device_id": fmt_ids("MD", [m["mdev"] for m in merchants]),
        "onboarding_ip_block": [ip_str(m["mip"]) for m in merchants],
        "website_present": [m["website_present"] for m in merchants],
        "website_category_match": pd.array([m["website_match"] for m in merchants], dtype="boolean"),
    })

    # ---- payers
    payers = pd.DataFrame({
        "payer_id": fmt_ids("P", np.arange(cfg.n_payers)),
        "payer_name": w.p_name,
        "home_city": np.array(CITIES, dtype=object)[w.p_city],
        "first_seen_date": w.p_first_seen.date,
        "primary_instrument_id": w.p_upi,
    })

    # ---- settlements
    S = {k: cat(st_parts, k) for k in ["merchant", "ts", "amount"]}
    settlements = pd.DataFrame({
        "settlement_id": fmt_ids("STL", np.arange(1, len(S["ts"]) + 1), 9),
        "merchant_id": mids[S["merchant"]],
        "settlement_timestamp": to_ts(S["ts"], w),
        "settled_amount": S["amount"].round(2),
    }).sort_values("settlement_timestamp", ignore_index=True)

    # ---- payouts
    P = {k: cat(po_parts, k) for k in ["merchant", "ts", "amount", "ben"]}
    if len(P["ts"]) > 0:
        payouts = pd.DataFrame({
            "payout_id": fmt_ids("PO", np.arange(1, len(P["ts"]) + 1), 9),
            "merchant_id": mids[P["merchant"]],
            "payout_timestamp": to_ts(P["ts"], w),
            "payout_amount": P["amount"].round(2),
            "beneficiary_id": fmt_ids("BEN", P["ben"]),
            "beneficiary_account_id": fmt_ids("BA", P["ben"]),
        }).sort_values("payout_timestamp", ignore_index=True)
    else:
        payouts = pd.DataFrame(columns=["payout_id", "merchant_id", "payout_timestamp", "payout_amount", "beneficiary_id", "beneficiary_account_id"])

    # ---- labels (ground truth only)
    rows = []
    lo, hi = cfg.label_delay_weeks
    for m in merchants:
        fs, known, src, obs = None, None, "none", 0
        if m["is_fraud"]:
            fs = w.start + pd.Timedelta(days=m["fraud_start"])
            if rng.random() < cfg.label_rate:
                known = fs + pd.Timedelta(days=int(rng.integers(lo * 7, hi * 7 + 1)))
                src, obs = str(rng.choice(["investigation", "user_report"], p=[.65, .35])), 1
            else:
                src = "never_labeled"
        elif rng.random() < cfg.false_label_rate:
            known = w.start + pd.Timedelta(days=int(rng.integers(30, w.n_days)))
            src, obs = "false_positive", 1
        rows.append(dict(
            merchant_id=m["merchant_id"], is_fraud=m["is_fraud"], archetype=m["archetype"],
            ring_id=m["ring"]["id"] if m["ring"] is not None else None,
            fraud_start_date=fs.date() if fs is not None else None,
            label_known_date=known.date() if known is not None else None,
            label_source=src, observed_label=obs))
    labels = pd.DataFrame(rows)

    config_df = pd.DataFrame([dict(
        run_id=f"run_seed{cfg.seed}", random_seed=cfg.seed,
        params_json=json.dumps(asdict(cfg)))])

    return dict(merchants=merchants_df, payers=payers, transactions=transactions,
                settlements=settlements, payouts=payouts,
                merchant_labels=labels, simulation_config=config_df)


# --------------------------------------------------------------------------- #
# Sanity report
# --------------------------------------------------------------------------- #
def auc(y, s):
    r = pd.Series(s).rank().to_numpy()
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def sanity_report(t):
    tx, lab = t["transactions"], t["merchant_labels"].set_index("merchant_id")
    col = tx[tx.txn_direction == "collection"].copy()
    hr = col.txn_timestamp.dt.hour
    col["round"] = (col.amount % 100 == 0)
    col["night"] = (hr >= 22) | (hr <= 3)
    g = col.groupby("merchant_id")
    f = pd.DataFrame({
        "round_share": g["round"].mean(),
        "night_share": g["night"].mean(),
        "n_payers": g["payer_id"].nunique(),
        "cb_rate": g["chargeback_flag"].mean(),
        "fail_rate": (g["status"].apply(lambda s: (s == "failed").mean()))
    })
    sett = t["settlements"].groupby("merchant_id")["settled_amount"].sum()
    outp = t["payouts"].groupby("merchant_id")["payout_amount"].sum() if len(t["payouts"]) > 0 else pd.Series(dtype=float)
    f["payout_ratio"] = (outp / sett).reindex(f.index).fillna(0)
    f = f.join(lab["is_fraud"])
    print("\n=== Sanity Report ===")
    print(f"merchants={len(lab)}  fraud={int(lab.is_fraud.sum())}  "
          f"transactions={len(tx):,}  payouts={len(t['payouts']):,}")
    print("Single-feature AUC vs is_fraud (ideal: between 0.55 and 0.88; none > 0.90):")
    for c in ["round_share", "night_share", "n_payers", "payout_ratio", "cb_rate", "fail_rate"]:
        a = auc(f.is_fraud, f[c])
        flag = "   <-- too easy, strengthen camouflage" if a > .9 else ("   <-- healthy" if a >= 0.55 else "")
        print(f"  {c:14s} {a:.3f}{flag}")
    print("\nArchetype comparison across key signals:")
    f2 = f.join(lab["archetype"]).groupby("archetype")[["round_share", "night_share", "payout_ratio", "cb_rate", "fail_rate"]].mean()
    print(f2.round(4))


def write_tables(tables, out):
    os.makedirs(out, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        ext = "parquet"
    except ImportError:
        ext = "csv"
    for name, df in tables.items():
        path = os.path.join(out, f"{name}.{ext}")
        df.to_parquet(path, index=False) if ext == "parquet" else df.to_csv(path, index=False)
        print(f"wrote {path}  ({len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merchants", type=int, default=1000)
    ap.add_argument("--weeks", type=int, default=52)
    ap.add_argument("--payers", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drift-week", type=int, default=30)
    ap.add_argument("--fraud-rate", type=float, default=0.03)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    cfg = Config(seed=a.seed, n_merchants=a.merchants, n_weeks=a.weeks, n_payers=a.payers,
                 drift_week=a.drift_week, fraud_rate=a.fraud_rate)
    tables = build_tables(cfg)
    write_tables(tables, a.out)
    sanity_report(tables)


if __name__ == "__main__":
    main()
