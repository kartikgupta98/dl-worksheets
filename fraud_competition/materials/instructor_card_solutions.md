# Technique card solutions (instructor only)

The change each card makes to the starter notebook, in the instructor guide's order, each on top of
the previous ones. Lines starting with `+` are added, `-` removed. For help in class: students do not
need to match this code, only the idea. `instructor_solution_notebook.ipynb` has every helpful card
applied; `card_test_results.csv` has the score each step reached when run.

## A2. Scale the inputs

**Imports**
```diff
@@ -5 +5,2 @@
 from sklearn.model_selection import train_test_split
+from sklearn.preprocessing import StandardScaler
```

**Train / validation split**
```diff
@@ -1 +1,6 @@
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
+
+scaler = StandardScaler().fit(X_train)
+X_train = scaler.transform(X_train)
+X_val = scaler.transform(X_val)
+X_test = scaler.transform(X_test)
```

## A1. Use more columns

**Features**
```diff
@@ -1,9 +1,2 @@
-FEATURES = [
-    "transaction_hour",
-    "is_international",
-    "failed_pin_attempts_24h",
-    "txn_count_last_1h",
-    "merchant_risk_score",
-    "device_trust_score",
-]
+FEATURES = train.select_dtypes("number").columns.drop("is_fraud").tolist()
 
```

## A3. Log-transform skewed columns

**Preprocessing**
```diff
@@ -1,2 +1,3 @@
 for df in [train, test]:
-    pass
+    for col in ["amount_inr", "customer_avg_spend_90d", "card_age_days", "distance_from_home_km", "txn_count_last_1h"]:
+        df[col] = np.log1p(df[col])
```

## A4. One-hot encode text columns

**Features**
```diff
@@ -2,4 +2,6 @@
 
-X = train[FEATURES].fillna(0)
-X_test = test[FEATURES].fillna(0)
+TEXT = ["merchant_category", "channel", "card_type", "city_tier"]
+X = pd.get_dummies(train[FEATURES + TEXT], columns=TEXT, dtype=float).fillna(0)
+X_test = pd.get_dummies(test[FEATURES + TEXT], columns=TEXT, dtype=float).fillna(0)
+X_test = X_test.reindex(columns=X.columns, fill_value=0)
 y = train["is_fraud"]
```

## A6. Fill blanks with the median + A7. Add "was missing" flags

**Preprocessing**
```diff
@@ -3 +3,3 @@
         df[col] = np.log1p(df[col])
+    for col in train.columns[train.isna().any()]:
+        df[col + "_missing"] = df[col].isna().astype(int)
```

**Features**
```diff
@@ -3,4 +3,4 @@
 TEXT = ["merchant_category", "channel", "card_type", "city_tier"]
-X = pd.get_dummies(train[FEATURES + TEXT], columns=TEXT, dtype=float).fillna(0)
-X_test = pd.get_dummies(test[FEATURES + TEXT], columns=TEXT, dtype=float).fillna(0)
+X = pd.get_dummies(train[FEATURES + TEXT], columns=TEXT, dtype=float)
+X_test = pd.get_dummies(test[FEATURES + TEXT], columns=TEXT, dtype=float)
 X_test = X_test.reindex(columns=X.columns, fill_value=0)
```

**Train / validation split**
```diff
@@ -2,2 +2,6 @@
 
+medians = X_train.median()
+X_train = X_train.fillna(medians)
+X_val = X_val.fillna(medians)
+X_test = X_test.fillna(medians)
 scaler = StandardScaler().fit(X_train)
```

## A8. Handle special codes

**Preprocessing**
```diff
@@ -1,2 +1,4 @@
 for df in [train, test]:
+    df["never_chargeback"] = (df["days_since_last_chargeback"] == -1).astype(int)
+    df["days_since_last_chargeback"] = np.log1p(df["days_since_last_chargeback"].replace(-1, np.nan))
     for col in ["amount_inr", "customer_avg_spend_90d", "card_age_days", "distance_from_home_km", "txn_count_last_1h"]:
```

## B1. Change the optimizer or learning rate

**Model**
```diff
@@ -7,3 +7,3 @@
 ])
-model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])
+model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
 model.summary()
```

## B2. Train for more epochs

**Training**
```diff
@@ -1,2 +1,2 @@
 history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
-                    epochs=10, batch_size=32, verbose=2)
+                    epochs=100, batch_size=32, verbose=2)
```

## C1. A wider or deeper network

**Model**
```diff
@@ -4,3 +4,4 @@
     keras.layers.Input(shape=(X_train.shape[1],)),
-    keras.layers.Dense(4, activation="relu"),
+    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dense(128, activation="relu"),
     keras.layers.Dense(1, activation="sigmoid"),
```

## D1. Dropout

**Model**
```diff
@@ -5,3 +5,5 @@
     keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dropout(0.3),
     keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dropout(0.3),
     keras.layers.Dense(1, activation="sigmoid"),
```

## A9. Drop columns that cannot carry signal

**Features**
```diff
@@ -1,2 +1,3 @@
 FEATURES = train.select_dtypes("number").columns.drop("is_fraud").tolist()
+FEATURES = [col for col in FEATURES if col not in ["batch_number", "acquirer_code", "pos_software_version"]]
 
```

## E1. Compare the amount with the customer's usual spend

**Preprocessing**
```diff
@@ -3,2 +3,3 @@
     df["days_since_last_chargeback"] = np.log1p(df["days_since_last_chargeback"].replace(-1, np.nan))
+    df["amount_vs_usual"] = np.log(df["amount_inr"] / df["customer_avg_spend_90d"])
     for col in ["amount_inr", "customer_avg_spend_90d", "card_age_days", "distance_from_home_km", "txn_count_last_1h"]:
```

## E2. Treat the hour as a circle

**Preprocessing**
```diff
@@ -4,2 +4,4 @@
     df["amount_vs_usual"] = np.log(df["amount_inr"] / df["customer_avg_spend_90d"])
+    df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
+    df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
     for col in ["amount_inr", "customer_avg_spend_90d", "card_age_days", "distance_from_home_km", "txn_count_last_1h"]:
```

## E3. Look at the amounts themselves

**Preprocessing**
```diff
@@ -4,2 +4,3 @@
     df["amount_vs_usual"] = np.log(df["amount_inr"] / df["customer_avg_spend_90d"])
+    df["amount_is_round"] = (df["amount_inr"] % 500 == 0).astype(int)
     df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
```

## B4. Lower the learning rate when progress stalls

**Training**
```diff
@@ -1,2 +1,3 @@
+reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
 history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
-                    epochs=100, batch_size=32, verbose=2)
+                    epochs=100, batch_size=32, callbacks=[reduce_lr], verbose=2)
```

## Cards that do not help, or replace another card

### A5. Label-encode text columns (on the full solution)

**Features**
```diff
@@ -4,5 +4,8 @@
 TEXT = ["merchant_category", "channel", "card_type", "city_tier"]
-X = pd.get_dummies(train[FEATURES + TEXT], columns=TEXT, dtype=float)
-X_test = pd.get_dummies(test[FEATURES + TEXT], columns=TEXT, dtype=float)
-X_test = X_test.reindex(columns=X.columns, fill_value=0)
+X = train[FEATURES + TEXT].copy()
+X_test = test[FEATURES + TEXT].copy()
+for col in TEXT:
+    categories = sorted(train[col].unique())
+    X[col] = pd.Categorical(X[col], categories=categories).codes
+    X_test[col] = pd.Categorical(X_test[col], categories=categories).codes
 y = train["is_fraud"]
```

### B3. Change the batch size (on the full solution)

**Training**
```diff
@@ -2,2 +2,2 @@
 history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
-                    epochs=100, batch_size=32, callbacks=[reduce_lr], verbose=2)
+                    epochs=100, batch_size=256, callbacks=[reduce_lr], verbose=2)
```

### C2. A very deep network (on the full solution)

**Model**
```diff
@@ -8,2 +8,8 @@
     keras.layers.Dropout(0.3),
+    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dropout(0.3),
+    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dropout(0.3),
+    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dropout(0.3),
     keras.layers.Dense(1, activation="sigmoid"),
```

### C3. A different activation (on the full solution)

**Model**
```diff
@@ -4,5 +4,5 @@
     keras.layers.Input(shape=(X_train.shape[1],)),
-    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dense(128, activation="sigmoid"),
     keras.layers.Dropout(0.3),
-    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dense(128, activation="sigmoid"),
     keras.layers.Dropout(0.3),
```

### C4. Batch normalization (on the full solution)

**Model**
```diff
@@ -5,4 +5,6 @@
     keras.layers.Dense(128, activation="relu"),
+    keras.layers.BatchNormalization(),
     keras.layers.Dropout(0.3),
     keras.layers.Dense(128, activation="relu"),
+    keras.layers.BatchNormalization(),
     keras.layers.Dropout(0.3),
```

### D2. L2 penalty (on step `L09 +2x128 network`)

**Model**
```diff
@@ -4,4 +4,4 @@
     keras.layers.Input(shape=(X_train.shape[1],)),
-    keras.layers.Dense(128, activation="relu"),
-    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.L2(1e-3)),
+    keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.L2(1e-3)),
     keras.layers.Dense(1, activation="sigmoid"),
```

### D3. L1 penalty (on the full solution)

**Model**
```diff
@@ -4,5 +4,5 @@
     keras.layers.Input(shape=(X_train.shape[1],)),
-    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.L1(1e-4)),
     keras.layers.Dropout(0.3),
-    keras.layers.Dense(128, activation="relu"),
+    keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.L1(1e-4)),
     keras.layers.Dropout(0.3),
```

### D4. Early stopping (on the full solution)

**Training**
```diff
@@ -1,3 +1,4 @@
+early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
 reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
 history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
-                    epochs=100, batch_size=32, callbacks=[reduce_lr], verbose=2)
+                    epochs=100, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=2)
```

### F1. Average several models (on the full solution)

**Submission**
```diff
@@ -1,2 +1,19 @@
-pred = (model.predict(X_test) > 0.5).astype(int).ravel()
+probs = []
+for seed in [0, 1, 2]:
+    keras.utils.set_random_seed(seed)
+
+    model = keras.Sequential([
+        keras.layers.Input(shape=(X_train.shape[1],)),
+        keras.layers.Dense(128, activation="relu"),
+        keras.layers.Dropout(0.3),
+        keras.layers.Dense(128, activation="relu"),
+        keras.layers.Dropout(0.3),
+        keras.layers.Dense(1, activation="sigmoid"),
+    ])
+    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
+    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
+    model.fit(X_train, y_train, validation_data=(X_val, y_val),
+              epochs=100, batch_size=32, callbacks=[reduce_lr], verbose=0)
+    probs.append(model.predict(X_test).ravel())
+pred = (np.mean(probs, axis=0) > 0.5).astype(int)
 submission = pd.DataFrame({"transaction_id": test["transaction_id"], "is_fraud": pred})
```
