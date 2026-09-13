# Card fraud competition: handoff notes

Status 2026-09-13: **calibrated, exported, materials written.** What remains is done by
hand on Kaggle (see `materials/kaggle_setup_checklist.md`). To resume in a new Claude Code
session: "Read fraud_competition/HANDOFF.md and continue."

## Goal

An in-class Kaggle Community Competition for students who have just learnt
underfitting / overfitting / L1 / L2 / dropout. Students get a messy tabular dataset and
build a full Keras pipeline: split, preprocess, size the network, read loss curves,
regularise, tune. Live leaderboard, ~2 hour session, metric = accuracy.

Design decisions agreed with the instructor:
- Simulated dataset presented as the case study of a fictional company (Lumen Pay, a placeholder name). No real brand names.
- About 10 visible improvement steps (at least ~1.5 points each).
- Target share of the gain: preprocessing ~45%, training ~15%, capacity + regularisation ~25%, feature engineering ~10% (must not dominate), tiebreakers ~3%.
- Each regularisation technique helps alone; stacking gives diminishing returns. Include cards that do not help.
- Keras neural networks only, low daily submission cap, 30% / 70% public / private split.

## Result (exp2, `results_exp2.csv`, full tables in `materials/instructor_guide.md`)

- Starter 59.7 → full pipeline 86.9 (3-seed average 87.3); best possible 92.0.
- Visible steps: scale +2.5, log +1.3 (+2.2 on the full pipeline), one-hot +5.8, median + missing flags +1.9, `-1` flag +1.6, Adam +3.2, 100 epochs +2.7, bigger network +1.5 (+6.4 if grown before training longer), dropout +3.4, round-amount flag +1.6, reduce LR +1.0.
- Shares (average of the two ladder orders): preprocessing 50%, training 16%, capacity + regularisation 27%, feature engineering 7%.
- On the overfitting 2×128 network: dropout +3.4 / +4.2, early stopping +2.2, L2 1e-3 +1.2; combinations +4.1 to +4.2.
- No-help cards (BatchNorm, sigmoid, 5 layers, batch 256, L1, early stopping on top of dropout) all within ±0.4; dropout on the small network −2.2.
- Card path checked in notebooks (`test_cards.py`, `card_test_results.csv`): the starter with the cards added one at a time, using the reference code, runs without errors and every score is within 0.8 points of the calibration run with the same seed (59.6 → 86.8; 87.3 with 3 averaged models).
- Technique cards are hints, not code: each says what to try, where in the notebook it goes, which functions to look up, what to do next and how to check it worked. The exact code is in `materials/instructor_card_solutions.md` (instructor only).
- Known weak spots: amount vs usual spend (+0.1) and hour sin/cos (+0.1) add almost nothing, because the network learns both from logged columns; the round-amount flag carries the feature-engineering share instead.

## How the generator was rebalanced (vs exp1)

exp1 (10k train, sharpness 3.6) had a dead starter (49.6), scaling +19, one-hot +12,
and no capacity gain (the true function was nearly additive). Changes, all as knobs in
`generate_data.py` whose defaults still reproduce the exp1 data exactly:
- 20k training rows; much weaker merchant / channel / card-type main effects.
- Informative blanks: location hidden (distance blank), spend profile reset (avg spend blank).
- Round cash-out amounts (multiples of 500) and whole-rupee amounts, so a hand-made flag is the one feature a network cannot learn.
- Night peak moved to about midnight; more extreme amounts (0.8%).
- 8 "sign flips" (`flips` knob): products of centred numeric facts such as card age × transactions in the last hour. These defeat a 4-unit network but not a regularised 2×128 one. Interactions tied to categories were avoided because they inflate one-hot.
- Starter: SGD on 6 small-valued numeric columns (`exp2.SMALL_NUMERIC`), so the first submission is alive (~60) and scaling / Adam stay as steps.

## Files

| File | Purpose |
|---|---|
| generate_data.py | Generator. Hidden clean facts → true fraud probability → label; released columns carry planted flaws. |
| pipeline.py | One Keras pipeline; every student step is a config switch. `run()` does the student-style 80/20 split and scores on the hidden test set. |
| exp2.py | Final settings (KNOBS, N_TRAIN, START) and grid: ladder (L), grow-first ladder (M), full-minus-one (F-), regularisation (R), no-help cards (N), starter variants (S), tiebreak (T). |
| calibrate.py | Parallel runner: `python calibrate.py exp2 8`. Appends to results_exp2.csv, skips finished jobs. |
| summary.py | `python summary.py exp2`: results with a gain column. |
| export_kaggle.py | `python export_kaggle.py` writes kaggle/ + private/solution.csv; `reference` writes private/reference_submission.csv; `score sub.csv` scores locally. Reads settings from exp2. |
| make_notebooks.py | Builds the notebooks from one description of the starter (exp2.START) plus a function per technique card: materials/starter_notebook.ipynb (students, Colab), materials/instructor_solution_notebook.ipynb (all helpful cards), materials/instructor_card_solutions.md (each card's exact change). |
| make_cards_html.py | `python make_cards_html.py`: builds materials/technique_cards.html (the student handout to share) from materials/technique_cards.md. Edit the .md, then rebuild. |
| test_cards.py | `python test_cards.py 8`: runs the notebook after every card on the ladder plus the no-help cards, scores each submission, compares with results_exp2.csv, writes card_test_results.csv. |
| smoke_test.py | Reproduces 3 recorded runs on a new machine. |
| exp1.py, results_exp1.csv | The first (unbalanced) calibration, kept for reference. |
| materials/ | competition_overview.md, data_description.md, technique_cards.md, instructor_guide.md, kaggle_setup_checklist.md, starter_notebook.ipynb |

## Running

```bash
cd dl-worksheets/fraud_competition
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python export_kaggle.py                # kaggle/ + private/; best possible 0.9222 public / 0.9190 private
python export_kaggle.py reference      # ~1 minute; scores 0.8721 public / 0.8662 private
python smoke_test.py                   # compares with results_exp2.csv
python make_notebooks.py               # rebuild the starter, solution notebook and card solutions
python test_cards.py 8                 # run every card's notebook and compare with the calibration
cp kaggle/train.csv kaggle/test.csv materials/   # the notebook reads them from its own folder, as in Colab
KERAS_BACKEND=jax jupyter nbconvert --to notebook --execute materials/starter_notebook.ipynb --output-dir /tmp
caffeinate -i python -u calibrate.py exp2 8 2>&1 | tee exp2.log   # full grid, 174 runs, ~11 minutes on 8 workers
python summary.py exp2
```

## Environment notes

- Keras 3 on the JAX backend by default (pipeline.py and calibrate.py set it), about 15× faster than torch on CPU for these small networks. Set `KERAS_BACKEND=torch` to use torch.
- One worker ≈ one CPU thread; 8 workers on a 10-core Mac.
- Students on Kaggle will use the TensorFlow backend; the notebook code is backend-neutral. Not yet checked on Kaggle itself.

## Next steps

0. **Everything above is in the public repo kartikgupta98/dl-worksheets** (pushed 2026-09-13 by choice), including exp2.py and the instructor materials. A student who finds it can regenerate the hidden test labels (exp2.KNOBS + seed 2026) and read the solutions; the Kaggle data and private/solution.csv are git-ignored and were not pushed. If that matters before class, make the repo private, or export with a new seed kept out of git (then rerun `calibrate.py`, `test_cards.py` and update the guide's numbers, which will shift slightly).
1. Check that "Lumen Pay" and "Flagged or Fraud?" do not collide with real brands.
2. Follow `materials/kaggle_setup_checklist.md`: create the competition, upload data and solution, starter notebook, TA dry run, time a 100-epoch run on Kaggle CPU.
3. Optional: print the technique cards (`materials/technique_cards.md`) as a handout.
