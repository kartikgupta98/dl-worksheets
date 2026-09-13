# Flagged or Fraud?

*Help Lumen Pay's investigators decide which flagged card transactions are really fraud.*

<!-- Paste into the Kaggle Overview tab. Lumen Pay is a fictional company. -->

## Description

Lumen Pay issues debit, credit and prepaid cards across India. Every card
transaction passes through an automated monitor, and anything that looks
unusual is **flagged** and sent to the fraud operations team.

Flagging is cautious by design, so the team is buried: roughly half of the
flagged transactions turn out to be genuine customers buying groceries at
midnight or topping up a wallet on holiday. Each investigation takes an
analyst about 20 minutes.

Your job is to build a neural network that looks at a flagged transaction and
predicts whether the investigation will confirm it as **fraud** (`is_fraud = 1`)
or clear it as **genuine** (`is_fraud = 0`).

The data comes straight from the payment system's export, and it has the
quirks you would expect from a production table. Read the Data tab carefully
before you model anything.

## Evaluation

Submissions are scored on **accuracy**: the fraction of test transactions whose
`is_fraud` you predict correctly.

The leaderboard you see during the competition is calculated on 30% of the test
set. Final standings use the other 70%, which nobody sees until the end. A model
tuned to squeeze the public leaderboard can drop on the private one, so trust
your own validation set.

## Submission file

For every `transaction_id` in `test.csv`, predict `is_fraud` as `0` or `1`
(not a probability). The file needs a header and exactly these two columns:

```
transaction_id,is_fraud
TXN00012345,0
TXN00098765,1
...
```

`sample_submission.csv` has the right format and predicts 0 for everything.

## Rules for this competition

1. **Keras neural networks only.** Your predictions must come from a Keras
   model. You may use pandas, NumPy and scikit-learn for preprocessing and
   splitting, but not scikit-learn (or any other library's) models for the
   predictions themselves.
2. **Individual competition.** Work on your own. Discuss ideas in the
   Discussion tab or out loud in class, but do not share code privately.
3. **Submission limit:** 8 submissions per day. Choose up to 2 final
   submissions for the private leaderboard; if you pick none, your best public
   scores are used.
4. **No external data** and no hand-labelling of the test set.

## Timeline

- **Start:** at the beginning of class
- **Final submission deadline:** end of class
- **Private leaderboard revealed:** straight after the deadline

## Getting started

Open the starter notebook in Google Colab, upload `train.csv` and `test.csv`,
run all cells, then download the `submission.csv` it writes and submit it here.
Then improve it one step at a time, checking your validation score and loss
curves after every change.
