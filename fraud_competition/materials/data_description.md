# Data

<!-- Paste into the Kaggle Data tab description. -->

Each row is one card transaction that Lumen Pay's monitor flagged for
investigation. The export was taken directly from the payment system.

## Files

| File | Rows | Contents |
|---|---|---|
| `train.csv` | 20,000 | Flagged transactions with the investigation outcome `is_fraud` |
| `test.csv` | 30,000 | Flagged transactions without `is_fraud`; predict these |
| `sample_submission.csv` | 30,000 | A correctly formatted submission (predicts 0 everywhere) |

## Columns

| Column | Type | Description |
|---|---|---|
| `transaction_id` | text | Unique ID of the transaction |
| `terminal_id` | text | ID of the terminal or payment gateway endpoint that processed it |
| `amount_inr` | number | Transaction amount in rupees |
| `customer_avg_spend_90d` | number | The cardholder's average transaction amount over the last 90 days. Blank when the customer has no 90-day spend profile |
| `card_age_days` | integer | Days since the card was issued |
| `transaction_hour` | integer | Local hour of the transaction, 0 to 23 |
| `merchant_category` | text | Type of merchant, e.g. `grocery`, `electronics`, `cash_withdrawal` |
| `channel` | text | `in_store` (chip and PIN), `contactless`, `online` or `atm` |
| `card_type` | text | `debit`, `credit` or `prepaid` |
| `city_tier` | text | Tier of the city where the card is registered: `tier_1`, `tier_2`, `tier_3` |
| `distance_from_home_km` | number | Distance between where the transaction happened and the cardholder's home address. Blank when the location could not be determined |
| `device_trust_score` | number | Device fingerprint trust score from 0 to 1 (higher is more trusted). Only online transactions are fingerprinted; blank when no fingerprint was read |
| `is_international` | 0 / 1 | 1 if the merchant is outside India |
| `failed_pin_attempts_24h` | integer | Wrong PIN entries on this card in the previous 24 hours |
| `txn_count_last_1h` | integer | Transactions on this card in the previous hour, including this one |
| `merchant_risk_score` | number | Risk score from 0 to 1 assigned to the merchant by the acquiring bank |
| `days_since_last_chargeback` | integer | Days since this card's most recent chargeback. `-1` means the card has never had a chargeback |
| `batch_number` | integer | Settlement batch the transaction was cleared in |
| `acquirer_code` | integer | Code of the acquiring bank |
| `pos_software_version` | number | Software version of the terminal |
| `is_fraud` | 0 / 1 | **Target.** 1 if the investigation confirmed fraud, 0 if the transaction was genuine. `train.csv` only |
