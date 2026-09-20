# Kaggle setup checklist (instructor)

Everything below is done by hand in the Kaggle website under your own account.

## Before creating the competition

- [ ] `python export_kaggle.py` has been run with the final settings; `kaggle/` and `private/` exist.
- [ ] `python export_kaggle.py score kaggle/sample_submission.csv` prints about 0.50 for both parts.
- [ ] `python export_kaggle.py reference` then `python export_kaggle.py score private/reference_submission.csv` prints about 0.87 for both parts.
- [ ] The company name (Lumen Pay) and competition title have been checked so they do not match a real brand.

## Create the Community Competition

- [ ] Kaggle → Competitions → Host a competition → create a new Community Competition.
- [ ] Title: *Flagged or Fraud?* Subtitle, thumbnail and URL slug set.
- [ ] Visibility / access: private or link-only, so only your class can join. Students must accept the rules to join.
- [ ] Overview tab: paste `materials/competition_overview.md` (description, evaluation, submission file, rules, timeline).

## Data and scoring

- [ ] Data tab: upload `kaggle/train.csv`, `kaggle/test.csv`, `kaggle/sample_submission.csv`.
- [ ] Data description: paste `materials/data_description.md`.
- [ ] Solution file: upload `private/solution.csv` (columns `transaction_id`, `is_fraud`, `Usage`, with `Usage` = `Public` / `Private`). **Never upload it to the Data tab.**
- [ ] Evaluation metric: accuracy on column `is_fraud`, ID column `transaction_id`.
- [ ] Sample submission set to `sample_submission.csv`, so Kaggle validates the format of every upload.
- [ ] Public / private split comes from the `Usage` column (30% / 70%).

## Limits and timeline

- [ ] Maximum daily submissions: 8.
- [ ] Maximum team size: 1.
- [ ] Final submissions a student may select: 2.
- [ ] Start and deadline set for the class slot. Kaggle deadlines are in **UTC** (IST = UTC + 5:30).

## Starter notebook

- [ ] Share `materials/starter_notebook.ipynb` with students (e.g. a Colab link or Drive file).
- [ ] Open it in Google Colab, upload `train.csv` and `test.csv` from the Kaggle Data tab to the Colab file panel, Run all.
- [ ] Download `submission.csv` from the Colab file panel and submit it: expect about 0.60 public (locally 0.602 public / 0.594 private).
- [ ] Time one 100-epoch run of the bigger network on Colab CPU (only measured locally so far). If it takes more than about 2 minutes, tell students to use early stopping or `batch_size=64`.

## Dry run with a TA account

- [ ] TA joins through the student invite link, accepts rules, sees all three data files.
- [ ] TA submits `sample_submission.csv` → about 0.50.
- [ ] TA submits a probability file (values between 0 and 1) → rejected or scored badly; decide whether to warn students.
- [ ] TA submits `private/reference_submission.csv` → public score matches the local `score` output (0.8721) to 4 decimals.
- [ ] TA selects final submissions; check the private leaderboard stays hidden until the deadline.
- [ ] Delete the TA's submissions or remove the TA from the leaderboard before class.

## In class

- [ ] Leaderboard page projected, auto-refresh.
- [ ] Students get only `materials/starter_notebook.ipynb` and the technique cards (`materials/technique_cards.html`).
- [ ] Instructor guide and `materials/instructor_*` files (card solutions, solution notebook) open for you and TAs, not shared.
- [ ] At the deadline, reveal the private leaderboard and discuss who moved and why.
