# Flagged or Fraud? Instructor guide

**Not for students.** This is the answer key: what is planted in the data, what each
technique card does, and the scores to expect.

All scores are accuracy on the hidden 30,000-row test set, mean of 3 seeds, from
`results_exp2.csv` (Keras 3, JAX backend, same split as the starter notebook). On
Kaggle's TensorFlow backend and with other seeds expect differences of about 0.5 to 1 point.

## At a glance

| Submission | Accuracy |
|---|---|
| `sample_submission.csv` (all 0) | 50.6 public / 50.3 private |
| Starter notebook, unchanged | 60.2 public / 59.4 private |
| Full reference pipeline (`private/reference_submission.csv`) | 87.2 public / 86.6 private |
| Best possible (true fraud probabilities known) | 92.2 public / 91.9 private |

A strong student finishes around 86 to 88. The public leaderboard has 9,038 rows, so
differences under about 0.5 point there are noise.

## What is planted in the data

| In the data | Fix | Card |
|---|---|---|
| `amount_inr`, `customer_avg_spend_90d`, `card_age_days`, `distance_from_home_km`, `txn_count_last_1h` have long tails; amounts reach 12.5 million | log1p, then scale | A3, A2 |
| Starter uses only 6 small numeric columns | add the rest, scaled and logged | A1 |
| 4 text columns (weak on their own, but channel matters inside interactions) | one-hot | A4 (A5 label codes is 1 point worse) |
| Blanks carry signal: distance is blank when location was hidden (risky); average spend is blank for new cards and for reset profiles (risky); device trust is blank for non-online and for blocked fingerprints (risky) | median fill **plus** missing flags | A6 + A7 |
| `days_since_last_chargeback = -1` means never, next to 0 = yesterday | flag and blank it | A8 |
| `batch_number`, `acquirer_code`, `pos_software_version`, IDs are pure noise | drop | A9 |
| Round cash-out amounts (multiples of 500): 9% of non-ATM rows, 74% of them fraud | a 0/1 column; a network cannot see divisibility in a scaled number | E3 |
| Sign flips between numeric facts, e.g. many transactions in an hour are risky on new cards but normal on old ones; spending above usual is risky on untrusted devices but normal on trusted ones (8 such pairs) | a bigger network, then regularisation | C1, D |
| Fraud peaks around midnight to 1 AM; amount relative to usual spend matters | present, but a network already learns both from the logged, scaled columns | E1, E2 add almost nothing |

## Suggested path and expected scores

Each row adds one card to everything above it.

| # | Step | Card | Test acc | Gain | Train acc |
|---|---|---|---|---|---|
| 0 | Starter: 6 numeric columns, blanks to 0, 1×4 network, SGD 0.01, 10 epochs | | 59.7 | | 58.9 |
| 1 | Scale inputs | A2 | 62.2 | +2.5 | 62.0 |
| 2 | All numeric columns | A1 | 62.4 | +0.2 | 61.9 |
| 3 | Log skewed columns | A3 | 63.7 | +1.3 | 63.2 |
| 4 | One-hot text columns | A4 | 69.5 | +5.8 | 68.9 |
| 5 | Median fill + missing flags | A6 + A7 | 71.4 | +1.9 | 71.2 |
| 6 | Chargeback `-1` flag | A8 | 73.0 | +1.6 | 72.6 |
| 7 | Adam, learning rate 1e-3 | B1 | 76.2 | +3.2 | 76.1 |
| 8 | 100 epochs | B2 | 78.9 | +2.7 | 79.4 |
| 9 | 2×128 network | C1 | 80.4 | +1.5 | **99.6** |
| 10 | Dropout 0.3 | D1 | 83.8 | +3.4 | 89.4 |
| 11 | Drop noise columns | A9 | 84.1 | +0.3 | 89.1 |
| 12 | Amount vs usual spend | E1 | 84.2 | +0.1 | 89.7 |
| 13 | Hour sin/cos | E2 | 84.3 | +0.1 | 89.8 |
| 14 | Round-amount flag | E3 | 85.9 | +1.6 | 91.1 |
| 15 | Reduce LR on plateau | B4 | 86.9 | +1.0 | 89.6 |

Adding all columns (step 2) only pays off once they are logged (step 3); together they are +1.5.

**The overfitting demo.** Swap steps 8 and 9 and the capacity lesson is unmistakable:

| After step 7 (76.2) | Test acc | Gain | Train acc |
|---|---|---|---|
| 2×128 network, still 10 epochs | 82.5 | **+6.4** | 88.6 |
| then 100 epochs | 80.4 | **−2.1** | 99.6 |
| then dropout 0.3 | 83.8 | +3.4 | 89.4 |

Show the loss curves of the middle row: validation loss turns upward while training loss keeps falling.

**Share of the total gain** (average of the two orders): preprocessing 50%, training
settings 16%, capacity + regularisation 27%, feature engineering 7%.

## Each step's value on the finished pipeline

Accuracy lost when one step is removed from the full pipeline (86.9).

| Removed | Loss |
|---|---|
| All numeric columns (back to the starter's 6) | 16.5 |
| Bigger network (and dropout on the 1×4 network) | 7.7 |
| Bigger network and dropout together | 4.5 |
| One-hot (text columns dropped) | 2.9 |
| Log skewed columns | 2.2 |
| Chargeback `-1` flag | 1.7 |
| Round-amount flag | 1.4 |
| Median fill + missing flags (back to 0) | 1.3 |
| Scaling | 1.2 |
| Dropout | 1.1 |
| One-hot replaced by label codes | 1.0 |
| Reduce LR on plateau | 1.0 |
| 100 epochs (back to 10) | 0.5 |
| Amount vs usual spend | 0.4 |
| Hour sin/cos | 0.2 |
| Adam (back to SGD 0.01, with 100 epochs) | 0.1 |
| Drop noise columns | 0.1 |

## Regularisation on the overfitting 2×128 network

After step 9: 80.4 test, 99.6 train. Each technique on its own:

| Technique | Test acc | Gain | Train acc | Epochs run |
|---|---|---|---|---|
| Dropout 0.3 | 83.8 | +3.4 | 89.4 | 100 |
| Dropout 0.5 | 84.6 | +4.2 | 85.9 | 100 |
| Early stopping (patience 15, restore best) | 82.6 | +2.2 | 94.9 | 23 |
| L2 1e-3 | 81.6 | +1.2 | 93.9 | 100 |
| L1 1e-4 | 81.2 | +0.8 | 97.5 | 100 |
| Drop noise columns | 81.2 | +0.8 | 99.6 | 100 |
| L2 1e-4 (too weak) | 80.7 | +0.3 | 99.0 | 100 |
| Dropout 0.3 + early stopping | 84.6 | +4.2 | 87.4 | 48 |
| Dropout 0.3 + L2 1e-4 + early stopping | 84.5 | +4.1 | 86.7 | 37 |

Every technique helps alone; stacking them gives diminishing returns.

## Cards that do not help

On the full pipeline (86.9):

| Card | Test acc | Change |
|---|---|---|
| BatchNorm after each dense layer | 86.9 | +0.1 |
| Sigmoid hidden activations | 86.8 | −0.1 |
| 5 hidden layers of 128 | 86.6 | −0.3 |
| Batch size 256 | 87.1 | +0.2 |
| Early stopping (with dropout already in) | 86.9 | 0.0 |
| L1 1e-4 (noise columns already dropped) | 87.3 | +0.4 |
| Average 3 seeds | 87.3 | +0.4 |

And one that hurts: **dropout 0.3 on the small 1×4 network** drops step 8 from 78.9 to
76.7 (−2.2). The model was underfitting, so regularising it makes things worse.

Starter variations: Adam instead of SGD in the starter gives +3.1; adding the unscaled
`card_age_days` to the starter makes it unstable (sd 2 points).

## Checked in notebooks

`python test_cards.py` adds the cards to the starter notebook one at a time, using the code in
`instructor_card_solutions.md`, runs each notebook the way Colab would and scores its
`submission.csv` on the full test set (seed 0). Every notebook ran without errors, and every
score is within 0.8 points of the calibration run with the same seed (`card_test_results.csv`).

| Step | Cards added | Notebook | Calibration (3-seed mean) |
|---|---|---|---|
| Starter | | 59.6 | 59.7 |
| Scale inputs | A2 | 63.2 | 62.2 |
| All numeric columns | A1 | 60.4 | 62.4 |
| Log skewed columns | A3 | 61.9 | 63.7 |
| One-hot | A4 | 69.2 | 69.5 |
| Median fill + missing flags | A6, A7 | 71.7 | 71.4 |
| Chargeback `-1` flag | A8 | 73.2 | 73.0 |
| Adam | B1 | 75.9 | 76.2 |
| 100 epochs | B2 | 78.1 | 78.9 |
| 2×128 network | C1 | 80.6 | 80.4 |
| Dropout 0.3 | D1 | 83.3 | 83.8 |
| Drop noise columns | A9 | 84.1 | 84.1 |
| Amount vs usual spend | E1 | 84.0 | 84.2 |
| Hour sin/cos | E2 | 84.4 | 84.3 |
| Round-amount flag | E3 | 85.8 | 85.9 |
| Reduce LR on plateau | B4 | 86.8 | 86.9 |

Other cards, on the full solution (86.8): label codes instead of one-hot (A5) 86.1, batch 256 (B3)
87.1, 5 layers (C2) 86.6, sigmoid (C3) 86.8, BatchNorm (C4) 86.9, L1 (D3) 87.2, early stopping (D4)
86.9, average 3 models (F1) 87.3. On the overfitting 2×128 network (80.6): L2 1e-3 (D2) 81.8, early
stopping (D4) 82.5. Dropout on the small network (78.1): 76.6.

**Expect a dip at steps 2 and 3.** With this seed, adding all columns drops 63.2 to 60.4 and the log
only brings it back to 61.9. The tiny 4-unit network is unstable there (about 2 points between
seeds); the gain becomes obvious at one-hot. If a student says "A1 made it worse", ask whether the
new columns are logged and scaled, and tell them to keep going.

A student's own code for a card will not match the reference line for line; expect about ±1 point
around these numbers, and more for a card that is only partly implemented.

## Common trouble and what to say

- **Stuck at about 50%, predicts one class.** An unscaled huge column (amounts reach 12.5 million). Scale, and log first.
- **Loss is `nan`.** Usually `np.log1p` on `days_since_last_chargeback` (−1 gives −inf), or `amount_vs_usual` for a blank average spend. Blank the −1 first; fill after feature engineering.
- **Shape mismatch at predict.** `get_dummies` on train and test separately produced different columns.
- **Median fill scores no better than 0.** The blanks are the signal; add the missing flags.
- **Big network, validation worse than the small one.** Look at train accuracy (near 100%): it is overfitting. That is the cue for section D.
- **Submission error.** Probabilities instead of 0/1, or a missing header.
- **Big public jumps.** Under 0.5 point on the public leaderboard is noise; trust validation, which tracks test within about 1 point.

## Two-hour run sheet

| Time | Activity |
|---|---|
| 0:00 to 0:10 | Story, join the competition, run and submit the starter (about 60) |
| 0:10 to 0:40 | Section A cards: scaling, columns, log, one-hot, blanks, special codes (about 73) |
| 0:40 to 1:05 | Sections B and C: Adam, epochs, a bigger network (about 80 to 82) |
| 1:05 to 1:30 | Section D: show overfitting curves, then dropout / early stopping / L2 (about 84) |
| 1:30 to 1:50 | Sections E and F, the no-help cards, select 2 final submissions (about 86 to 87) |
| 1:50 to 2:00 | Reveal the private leaderboard; discuss who moved and why |

## Files

| File | Use |
|---|---|
| `kaggle/train.csv`, `test.csv`, `sample_submission.csv` | Kaggle Data tab |
| `private/solution.csv` | Kaggle hidden solution. Never share |
| `private/reference_submission.csv` | Dry-run check of the scorer |
| `materials/competition_overview.md` | Kaggle Overview tab |
| `materials/data_description.md` | Kaggle Data tab description |
| `materials/starter_notebook.ipynb` | Starter notebook for Google Colab (reads `train.csv` / `test.csv` from the working folder) |
| `materials/technique_cards.html` | Student handout: what to try, where it goes, which functions to look up (built from `technique_cards.md` by `make_cards_html.py`) |
| `materials/instructor_card_solutions.md` | The exact change each card makes to the notebook, for helping stuck students |
| `materials/instructor_solution_notebook.ipynb` | Starter with every helpful card applied (~87) |
| `card_test_results.csv` | Scores of the notebooks after each card |
| `materials/kaggle_setup_checklist.md` | Setting up and dry-running the competition |
| `results_exp2.csv`, `exp2.py` | The calibration behind every number above |
