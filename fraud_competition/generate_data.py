"""
Simulated card fraud investigation dataset.

Every transaction has a hidden, clean description (how unusual the amount is
for this customer, how risky the merchant is, whether it happened at night, ...).
Those clean facts decide the true probability that the transaction is fraud,
and the label is drawn from that probability.

The released columns are then written the way a payment system would store
them: raw rupees with extreme values, text categories, missing readings, a -1
code for "never", and a few columns that carry no signal at all. Each
preprocessing step a student applies undoes one of these, which is why the
gains are real.
"""
import numpy as np
import pandas as pd

# frequency weight, risk (added to the log-odds of fraud)
MERCHANTS = {
    "grocery":          (14, -1.1),
    "fuel":             (8, -0.8),
    "restaurants":      (12, -0.6),
    "pharmacy":         (5, -0.9),
    "utilities":        (6, -1.0),
    "apparel":          (9, -0.1),
    "entertainment":    (6, -0.2),
    "home_improvement": (4, -0.3),
    "travel":           (6, 0.4),
    "electronics":      (8, 0.9),
    "jewellery":        (3, 0.8),
    "gaming":           (5, 0.6),
    "gift_cards":       (4, 1.3),
    "money_transfer":   (5, 1.0),
    "crypto_exchange":  (3, 1.4),
}
PHYSICAL = {"grocery", "fuel", "restaurants", "pharmacy", "apparel", "home_improvement", "jewellery"}
HIGH_RISK = {"electronics", "jewellery", "gift_cards", "money_transfer", "crypto_exchange"}

DEFAULT_KNOBS = dict(
    merchant=1.0,
    channel={"in_store": -0.5, "contactless": -0.3, "online": 0.4, "atm": 0.2},
    card_type={"debit": 0.0, "credit": 0.15, "prepaid": 0.7},
    city_tier={"tier_1": 0.1, "tier_2": 0.0, "tier_3": -0.1},
    new_card=1.0,          # bonus for very new cards, fades over ~2 months
    amount_above_usual=1.0,  # per unit of log(amount / usual spend) above usual
    amount_tiny=0.8,       # far below usual spend (card testing)
    night=1.5,             # peak effect around 2 AM
    international=0.8,
    distance_in_person=0.9,  # card used in person far from home (cloned card)
    distance_online=0.1,
    device_blocked=1.2,    # online, fingerprint blocked
    device_low_trust=2.0,  # online, trust score well below normal
    failed_pin=0.7,
    velocity=0.8,
    merchant_risk_score=0.5,
    recent_chargeback=2.0,  # fades over ~45 days
    never_chargeback=-0.5,
    new_card_online=1.2,
    big_amount_risky_merchant=0.7,
    night_risky_merchant=0.8,
    night_international=0.8,
    sharpness=1.0,         # scales all log-odds; controls the accuracy ceiling
    outlier_rate=0.004,
    distance_missing=0.10,
    device_blocked_rate=0.25,
    chargeback_rate=0.30,
)


def _hours(rng, n):
    # genuine card activity: quiet at night, busy from late morning to evening
    w = np.array([2, 1.2, 0.8, 0.6, 0.6, 1, 2, 3.5, 5, 6, 7, 7.5,
                  8, 7.5, 7, 7, 7.5, 8, 8.5, 8.5, 8, 7, 5, 3.5])
    return rng.choice(24, n, p=w / w.sum())


def generate(n, seed, knobs=None):
    k = dict(DEFAULT_KNOBS)
    if knobs:
        k.update(knobs)
    rng = np.random.default_rng(seed)

    names = list(MERCHANTS)
    weights = np.array([MERCHANTS[m][0] for m in names], float)
    merchant = rng.choice(names, n, p=weights / weights.sum())
    merchant_risk = np.array([MERCHANTS[m][1] for m in merchant])
    physical = np.isin(merchant, list(PHYSICAL))
    high_risk = np.isin(merchant, list(HIGH_RISK))

    u = rng.random(n)
    channel = np.where(physical,
                       np.select([u < 0.55, u < 0.85], ["in_store", "contactless"], "online"),
                       np.select([u < 0.15, u < 0.25], ["in_store", "contactless"], "online"))
    atm = rng.random(n) < 0.07
    channel = np.where(atm, "atm", channel)
    merchant = np.where(atm, "cash_withdrawal", merchant)
    merchant_risk = np.where(atm, 0.0, merchant_risk)
    high_risk = high_risk & ~atm
    online = channel == "online"

    card_type = rng.choice(["debit", "credit", "prepaid"], n, p=[0.55, 0.38, 0.07])
    city_tier = rng.choice(["tier_1", "tier_2", "tier_3"], n, p=[0.45, 0.35, 0.20])

    card_age = np.clip(np.exp(rng.normal(5.7, 1.2, n)), 1, 4000).astype(int)
    usual_spend = np.exp(rng.normal(7.7, 1.0, n))
    log_ratio = rng.normal(0, 0.8, n)
    amount = usual_spend * np.exp(log_ratio)
    outlier = rng.random(n) < k["outlier_rate"]
    amount = np.where(outlier, amount * np.exp(rng.uniform(3, 6, n)), amount)
    amount = np.where(atm, np.maximum(100, np.round(amount / 100) * 100), np.round(amount, 2))

    hour = _hours(rng, n)
    night = ((np.cos(2 * np.pi * (hour - 2) / 24) + 1) / 2) ** 3

    international = rng.random(n) < np.where(online, 0.09, 0.02)
    distance = np.where(online, np.exp(rng.normal(3.0, 1.6, n)), np.exp(rng.normal(1.3, 1.0, n)))
    distance = np.where(international, distance + np.exp(rng.normal(7.5, 0.5, n)), distance)
    distance = np.round(distance, 1)

    blocked = online & (rng.random(n) < k["device_blocked_rate"])
    trust = np.round(rng.beta(5, 2, n), 3)

    failed_pin = np.where(online, 0, np.minimum(rng.poisson(0.12, n), 5))
    velocity = np.minimum(1 + rng.negative_binomial(1, 0.6, n), 40)
    merchant_risk_score = np.round(rng.beta(2, 5, n), 3)

    has_chargeback = rng.random(n) < k["chargeback_rate"]
    chargeback_days = np.clip(np.exp(rng.normal(4.8, 1.2, n)), 0, 2000).astype(int)

    # ---- true log-odds of fraud, from the clean facts ----
    logit = (
        k["merchant"] * merchant_risk
        + np.vectorize(k["channel"].get)(channel)
        + np.vectorize(k["card_type"].get)(card_type)
        + np.vectorize(k["city_tier"].get)(city_tier)
        + k["new_card"] * np.exp(-card_age / 60)
        + k["amount_above_usual"] * np.clip(log_ratio, 0, 2.5)
        + k["amount_tiny"] * (log_ratio < -1.0)
        + k["night"] * night
        + k["international"] * international
        + np.where(online, k["distance_online"], k["distance_in_person"])
        * np.clip((np.log1p(distance) - 2.0) / 2, 0, 3)
        + np.where(online, np.where(blocked, k["device_blocked"],
                                    k["device_low_trust"] / (1 + np.exp(-(0.5 - trust) * 15))), 0)
        + k["failed_pin"] * np.minimum(failed_pin, 3)
        + k["velocity"] * np.log1p(velocity - 1)
        + k["merchant_risk_score"] * (merchant_risk_score - 0.28) / 0.16
        + np.where(has_chargeback, k["recent_chargeback"] * np.exp(-chargeback_days / 45), k["never_chargeback"])
        + k["new_card_online"] * (online & (card_age < 90))
        + k["big_amount_risky_merchant"] * np.maximum(log_ratio, 0) * high_risk
        + k["night_risky_merchant"] * night * high_risk
        + k["night_international"] * night * international
    ) * k["sharpness"]

    # choose the intercept so that half the investigated transactions are fraud
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() < 0.5:
            lo = mid
        else:
            hi = mid
    p = 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))
    is_fraud = (rng.random(n) < p).astype(int)

    # ---- released columns, stored the way the payment system stores them ----
    df = pd.DataFrame({
        "transaction_id": [f"TXN{v:08d}" for v in rng.choice(10**8, n, replace=False)],
        "terminal_id": [f"T{v:05d}" for v in rng.integers(10000, 12500, n)],
        "amount_inr": amount,
        "customer_avg_spend_90d": np.where(card_age < 90, np.nan, np.round(usual_spend, 2)),
        "card_age_days": card_age,
        "transaction_hour": hour,
        "merchant_category": merchant,
        "channel": channel,
        "card_type": card_type,
        "city_tier": city_tier,
        "distance_from_home_km": np.where(rng.random(n) < k["distance_missing"], np.nan, distance),
        "device_trust_score": np.where(online & ~blocked, trust, np.nan),
        "is_international": international.astype(int),
        "failed_pin_attempts_24h": failed_pin,
        "txn_count_last_1h": velocity,
        "merchant_risk_score": merchant_risk_score,
        "days_since_last_chargeback": np.where(has_chargeback, chargeback_days, -1),
        "batch_number": rng.integers(1, 500, n),
        "acquirer_code": rng.choice(np.arange(1101, 1125), n),
        "pos_software_version": rng.choice([3.1, 3.4, 4.0, 4.2, 4.5], n),
        "is_fraud": is_fraud,
    })
    return df, p


if __name__ == "__main__":
    df, p = generate(40_000, seed=2026)
    print(df.head())
    print("fraud rate", df.is_fraud.mean())
    print("best possible accuracy", np.maximum(p, 1 - p).mean())
    print(df.isna().mean().round(3))
