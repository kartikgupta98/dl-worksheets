# Card fraud competition: handoff notes

Paused 2026-09-13 in the middle of calibration. To resume in a new Claude Code session:
"Read fraud_competition/HANDOFF.md and continue."

## Goal

An in-class Kaggle Community Competition for students who have just learnt
underfitting / overfitting / L1 / L2 / dropout (see DL_Overfitting_Underfitting_Simple.ipynb).
Students get a messy tabular dataset and build a full Keras pipeline: split,
preprocess, size the network, read loss curves, regularise, tune. Live
leaderboard projected in class, ~2 hour session, metric = accuracy.

Design decisions agreed with the instructor:
- Simulated dataset, presented as a real case study of a fictional company (no
  need to tell students it is simulated). No real brand names.
- Story: a fintech fraud team investigates flagged card transactions; predict
  which are confirmed fraud. About 50/50 classes.
- About 10 steps of improvement, each large enough to see (at least ~1.5 points),
  so the class fills ~2 hours and the leaderboard spreads out.
- Target share of the total gain: preprocessing ~45%, training settings ~15%,
  capacity + regularisation ~25%, feature engineering ~10% (must NOT dominate),
  tiebreakers ~3%.
- Each regularisation technique (dropout, L2, early stopping) should help on its
  own; stacking them gives diminishing returns.
- Include technique cards that do not help (BatchNorm after scaling, sigmoid
  activations, 5+ layers, batch size, L1 once ID columns are gone, dropout on the
  small model).
- Rule: Keras neural networks only. Low daily submission cap. Public/private
  leaderboard split (30% / 70%), final ranking on private.
- Train ~10k rows, test 30k rows. Best possible accuracy ~92%.

## Files

| File | Purpose |
|---|---|
| generate_data.py | Generator. Hidden clean facts -> true fraud probability -> label; released columns carry planted flaws. Knobs in DEFAULT_KNOBS. |
| pipeline.py | One Keras pipeline; every student step is a config switch. STARTER = the starter notebook settings. `run()` does student-style 80/20 split and scores on the hidden test set. |
| calibrate.py | Parallel runner: `python calibrate.py <exp_module> <workers>`. Appends to results_<exp>.csv, skips finished jobs on restart. |
| exp1.py | First grid: cumulative ladder, full-minus-one-step, regularisation options, starter variants, null cards. KNOBS = {"sharpness": 3.6}. |
| results_exp1.csv | 35 of 141 runs finished before pausing. |
| export_kaggle.py | Writes kaggle/ (train, test, sample_submission) and private/solution.csv (Usage Public/Private). `python export_kaggle.py score sub.csv` scores locally. KNOBS must be updated to the final calibration. |

Planted flaws in the released columns:
- amount_inr: raw rupees, very skewed, 0.4% extreme outliers
- customer_avg_spend_90d: missing for cards < 90 days old
- merchant_category / channel / card_type / city_tier: text
- device_trust_score: NaN for non-online, and NaN when fingerprint blocked (blocked is risky)
- distance_from_home_km: missing at random 10%
- days_since_last_chargeback: -1 = never (sits next to 0 = yesterday)
- transaction_hour: fraud peaks ~2 AM (cyclic)
- batch_number, acquirer_code, pos_software_version: pure noise; transaction_id, terminal_id: string IDs
- Engineered features that help: log(amount / avg spend), hour sin/cos
- Interactions: new card x online, amount above usual x risky merchant, night x risky merchant, night x international, in-person x far from home

## Running on the Mac

```bash
git clone https://github.com/kartikgupta98/dl-worksheets.git
cd dl-worksheets
git checkout fraud-competition
cd fraud_competition
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python export_kaggle.py          # writes kaggle/ + private/, best possible accuracy public 0.9198, private 0.9213
python smoke_test.py             # L00 0.4959, L01 ~0.685, L06 ~0.867
sysctl -n hw.perflevel0.physicalcpu   # performance cores = max workers
caffeinate -i python -u calibrate.py exp1 8 2>&1 | tee exp1.log   # resumes, skips the 35 done runs
python summary.py exp1
```

## Environment notes

- Keras 3 on the torch backend (pipeline.py sets KERAS_BACKEND=torch by default;
  TensorFlow also works if installed). `pip install keras scikit-learn pandas numpy torch`.
- Each worker needs ~400 MB RAM; 14 workers crashed a 16 GB laptop with ~5 GB free, 8 worked.
- One 100-epoch run at batch 32 takes ~110 s single-threaded.

## Results so far (sharpness 3.6, best possible accuracy 92.1%, mean of 3 seeds)

| Ladder step (cumulative) | Test acc | Gain | Verdict |
|---|---|---|---|
| L00 starter (numeric only, NaN->0, no scaling, 1x4 units, SGD 0.01, 10 epochs) | 49.6 | | dead: predicts one class |
| L01 + StandardScaler | 69.0 | +19.4 | far too big |
| L02 + log1p skewed | 70.8 | +1.8 | ok |
| L03 + one-hot | 82.8 | +12.0 | far too big |
| L04 + median fill + missing flags | 83.8 | +1.0 | too small |
| L05 + chargeback -1 flag | 85.2 | +1.5 | ok-ish |
| L06 + Adam 1e-3 | 86.7 | +1.4 | ok-ish |
| L07 + 100 epochs | 87.5 | +0.8 | too small |
| L08 + 2x128 network | 86.7 | -0.8 | overfits immediately (train 99.4%); capacity gain invisible |
| L09 + dropout 0.3 | 88.1 | +1.4 | ok |
| L10 + drop noise columns | 88.3 | +0.2 | too small |
| L11 + amount vs usual spend | 88.6 | +0.3 | too small |

Not yet measured: L12-L13, all full-minus-one-step configs, regularisation
options, starter variants, null cards.

Earlier sklearn check at sharpness 3.6: logistic regression on fully cleaned
features 88.0%, sklearn MLP 89.5-90.1%, so the true function is still too
close to linear.

## Next steps

1. Fix the dead starter so its first submission is not ~50% and scaling becomes
   a normal-sized step. Ideas to test: starter with Adam instead of SGD
   (then the training step becomes learning rate / epochs); a starter that uses
   a hand-picked subset of numeric columns ("use all columns" becomes a step).
2. Rebalance the generator:
   - weaken merchant_category and channel effects (one-hot gain 12 -> ~4)
   - strengthen informative missingness (device blocked, avg spend missing for new cards)
     so missing handling gives ~3
   - make the true function more non-linear / interaction-heavy so a 4-unit
     network clearly underfits (capacity gain ~4), and consider a larger train
     set so the big network does not overfit instantly
   - strengthen the amount-vs-usual and night effects a little so feature engineering gives ~3 total
3. Re-run the grid (new exp2.py), iterate until every step clears ~1.5 points
   in both the ladder and full-minus-one views and shares match the targets.
4. Set final KNOBS in export_kaggle.py and export.
5. Build the starter notebook, the technique sheet (help + no-help cards,
   pitfalls), the competition brief / data description, and the Kaggle setup
   checklist (invite-only, accuracy metric, daily cap, team size, 30/70 split,
   dry run with a TA account).
