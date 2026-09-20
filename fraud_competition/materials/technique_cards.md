# Technique cards

Each card is one change you can try. **Not every card helps on this data**, and some
help only after others. Change one thing, re-run the whole notebook
(Runtime → Run all), and compare the validation accuracy and the curves with your
previous run before keeping it.

## Where changes go

Keep your notebook in this order:

1. **Preprocessing:** the `for df in [train, test]:` loop. Anything that creates or changes a column goes here, so train and test get exactly the same change.
2. **Features:** which columns go into `X` and `X_test`, and how blanks are filled.
3. **Right after the split:** anything that learns numbers from the data (medians, a scaler). Learn them from `X_train` only, then apply them to `X_train`, `X_val` and `X_test`.
4. **Model:** the layers and `compile`.
5. **Training:** `model.fit(...)`.
6. **Submission:** the prediction and `submission.csv`.

Inside the Preprocessing loop, create new columns from raw values first, then transform
existing columns (for example with log), then add flags for blanks.

---

## A. Preparing the data

### A1. Use more columns
**Idea:** The starter uses six columns. The table has many more.
**Where:** Features.
**How:** Instead of typing the names, build `FEATURES` from every numeric column of `train` except `is_fraud`. Look up `train.select_dtypes(...)` and `.columns.drop(...)`.
**Then:** Look at `train.describe()`. Are the new columns on very different scales from the old ones?
**Check:** `len(FEATURES)` has grown. Because `FEATURES` is built after the Preprocessing loop, numeric columns you create there later are picked up automatically.

### A2. Scale the inputs
**Idea:** A network learns badly when one input is in lakhs and another is between 0 and 1.
**Where:** Right after the split.
**How:** Use `StandardScaler` from `sklearn.preprocessing`. `fit` it on `X_train` only, then `transform` each of `X_train`, `X_val` and `X_test`.
**Check:** `X_train.mean(axis=0)` is close to 0 and `X_train.std(axis=0)` close to 1 for every column.

### A3. Log-transform skewed columns
**Idea:** Money, counts and distances often have a long tail: most values are small and a few are huge. A log squashes the tail.
**Where:** Preprocessing loop.
**How:** In `train.describe()`, find the columns whose `max` is far larger than their `75%` value. Replace each of them with `np.log1p(...)` of itself (`log1p` also works for 0).
**Check:** In `train.describe()`, the `max` of those columns is now only a few times the median.
**Watch:** `np.log1p` of a negative number gives `nan` or `-inf`. One skewed column uses `-1` as a code: read A8 before logging it.

### A4. One-hot encode text columns
**Idea:** `merchant_category`, `channel`, `card_type` and `city_tier` are text, so the starter ignores them. One-hot encoding turns each category into its own 0/1 column.
**Where:** Features, where `X` and `X_test` are built.
**How:** Select the text columns together with `FEATURES`, and pass the frame to `pd.get_dummies(..., columns=[...], dtype=float)`. Do it for `train` and for `test`.
**Then:** Train and test can end up with different dummy columns, or the same ones in a different order. Make `X_test` have exactly the columns of `X`: look up `DataFrame.reindex(columns=..., fill_value=0)`.
**Check:** `list(X.columns) == list(X_test.columns)` is `True`.

### A5. Label-encode text columns
**Idea:** Replace each category with a number (`grocery` = 0, `fuel` = 1, ...). Fewer columns than one-hot, but the numbers suggest an order that does not exist.
**Where:** Features.
**How:** For each text column, take the sorted list of categories from `train`. Convert the column in both `X` and `X_test` with that same list: `pd.Categorical(values, categories=...)`, then `.codes`.
**Watch:** Converting train and test separately can give the same category different numbers.

### A6. Fill blanks with the median
**Idea:** Filling with 0 claims the value was 0. The median is a typical value.
**Where:** Right after the split, before any scaling.
**How:** First remove `.fillna(0)` from the Features cell, otherwise there is nothing left to fill. Compute the medians of `X_train` with `X_train.median()`, and use them to `fillna` all three of `X_train`, `X_val` and `X_test`.
**Check:** `X_train.isna().sum().sum()` is 0, and the same for `X_val` and `X_test`.

### A7. Add "was missing" flags
**Idea:** Sometimes the fact that a value is blank is itself a clue. Filling the blank loses that clue unless you keep a flag.
**Where:** Preprocessing loop, after the other column changes (and before any filling).
**How:** Find the columns with blanks with `train.isna().sum()`. For each, add a column such as `<column>_missing` that is 1 where the value is blank and 0 otherwise: `.isna()`, then `.astype(int)`.
**Then:** Add the new columns to `FEATURES` (A1's automatic list already includes them).
**Check:** The mean of each new column equals that column's share of blanks.

### A8. Handle special codes
**Idea:** In `days_since_last_chargeback`, `-1` means "never". To a network, `-1` just looks like a day before `0`.
**Where:** Preprocessing loop, before any log.
**How:** Add a 0/1 column that marks the rows with `-1`. Then `replace` the `-1` with a blank (`np.nan`), so the rest of the column is a real number of days. It is a long-tailed column too, so it can now be logged like A3.
**Then:** Add the new column to `FEATURES`. The new blanks are filled wherever you fill blanks.
**Check:** `(train["days_since_last_chargeback"] == -1).sum()` is 0.

### A9. Drop columns that cannot carry signal
**Idea:** IDs and purely administrative columns can only add noise.
**Where:** Features.
**How:** Read the Data tab and decide which numeric columns describe bookkeeping rather than the transaction. Remove them from `FEATURES`, for example with a list comprehension.

---

## B. Training settings

### B1. Change the optimizer or learning rate
**Where:** Model, in `compile`.
**How:** Try `keras.optimizers.Adam` instead of `SGD`. `learning_rate=1e-3` is the usual starting point; try values ten times bigger or smaller.

### B2. Train for more epochs
**Where:** Training.
**How:** Increase `epochs`.
**Check:** In the curves, was the validation loss still going down at the last epoch?

### B3. Change the batch size
**Where:** Training.
**How:** Change `batch_size`, e.g. 64 or 256.

### B4. Lower the learning rate when progress stalls
**Where:** Training.
**How:** Create a `keras.callbacks.ReduceLROnPlateau` that watches `"val_loss"` (try `factor=0.5`, `patience=5`) and pass it to `model.fit` with `callbacks=[...]`.
**Check:** Give the callback `verbose=1` and it prints a line each time it lowers the rate.

---

## C. Model capacity

### C1. A wider or deeper network
**Where:** Model.
**How:** More units in the hidden `Dense` layer, or more `Dense` layers, e.g. two layers of 128.
**Check:** Compare train and validation accuracy in the curves. A growing gap means the model is memorising the training rows.

### C2. A very deep network
**Where:** Model.
**How:** Five or more hidden `Dense` layers.

### C3. A different activation
**Where:** Model.
**How:** `activation="sigmoid"` or `"tanh"` in the hidden layers instead of `"relu"`.

### C4. Batch normalization
**Where:** Model.
**How:** Add `keras.layers.BatchNormalization()` after each hidden `Dense` layer.

---

## D. Regularisation

### D1. Dropout
**Where:** Model.
**How:** Add `keras.layers.Dropout(rate)` after each hidden `Dense` layer. Rates between 0.2 and 0.5 are common.
**Check:** Train accuracy drops a little; validation accuracy should not.

### D2. L2 penalty
**Where:** Model.
**How:** Give each hidden `Dense` layer `kernel_regularizer=keras.regularizers.L2(strength)`. The strength matters: try values ten times apart, such as `1e-4`, `1e-3`, `1e-2`.

### D3. L1 penalty
**Where:** Model.
**How:** As D2, with `keras.regularizers.L1(strength)`.

### D4. Early stopping
**Where:** Training.
**How:** Create a `keras.callbacks.EarlyStopping` that watches `"val_loss"`, with a `patience` (e.g. 15) and `restore_best_weights=True`, and pass it in `callbacks=[...]`. Several callbacks go in the same list. Keep `epochs` high; training stops by itself.
**Check:** The training log ends before the last epoch.

---

## E. Feature engineering

### E1. Compare the amount with the customer's usual spend
**Idea:** ₹20,000 is normal for one customer and alarming for another.
**Where:** Preprocessing loop, before any log of the amount columns (it needs raw rupees).
**How:** Add a column: `amount_inr` divided by `customer_avg_spend_90d`. Take `np.log` of the ratio, so "twice as usual" and "half as usual" are the same distance from 1.
**Then:** Add it to `FEATURES`. A blank average spend gives a blank ratio; fill it like the other blanks.

### E2. Treat the hour as a circle
**Idea:** Hour 23 and hour 0 are one hour apart, but 23 apart as numbers.
**Where:** Preprocessing loop.
**How:** Add two columns: the sine and the cosine of 2π × hour / 24 (`np.sin`, `np.cos`, `np.pi`).
**Then:** Add them to `FEATURES`.
**Check:** Hours 23 and 0 now have almost the same pair of values.

### E3. Look at the amounts themselves
**Idea:** Some amounts may have been typed in by a person rather than rung up at a till.
**Where:** Preprocessing loop, before any log of `amount_inr`.
**How:** Print a few dozen `amount_inr` values from fraud rows and from genuine rows (`train[train["is_fraud"] == 1]`). If you spot a pattern, turn it into a 0/1 column. The `%` operator gives the remainder of a division.
**Then:** Add it to `FEATURES`.
**Check:** `train.groupby("<your column>")["is_fraud"].mean()` shows whether the pattern is linked to fraud.

---

## F. Final polish

### F1. Average several models
**Idea:** Networks started from different random weights make slightly different mistakes; averaging them cancels some out.
**Where:** Submission, replacing the prediction line.
**How:** Loop over a few seeds. In each pass: set the seed, build and compile a fresh model, `fit` it, and keep `model.predict(X_test)`. Average the kept probabilities with `np.mean(..., axis=0)`, then threshold at 0.5.
**Check:** `submission["is_fraud"]` still contains only 0 and 1.

---

## Pitfalls

- **Changing several things at once.** You will not know which change helped.
- **Running cells out of order.** The Preprocessing loop changes `train` and `test` in place: running it twice logs columns twice. After editing it, use Runtime → Run all.
- **Chasing the public leaderboard.** It uses 30% of the test set; final ranks use the other 70%. Trust your validation split.
- **Loss is `nan`.** Usually an unscaled huge column, a log of a negative number, or blanks left unfilled.
- **Validation accuracy stuck at about 50%.** The network predicts one class for everything. Check the inputs.
- **Error at `model.predict(X_test)`.** `X_test` does not have the same columns as `X`.
- **Submitting probabilities.** `is_fraud` must be 0 or 1.
