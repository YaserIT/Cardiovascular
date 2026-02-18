# ======================================================
# 0. Imports
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)
tf.random.set_seed(42)

# ======================================================
# 1. Parameters
# ======================================================
K = 20                  # 🔹 Number of full experiment runs
NUM_POINTS = 20
ratios = np.linspace(0.05, 1.0, NUM_POINTS)
metrics = ["accuracy", "precision", "recall", "f1", "auc", "latency"]

# ======================================================
# 2. Load & preprocess dataset
# ======================================================
data = pd.read_csv("heart.csv")
data = data.select_dtypes(include=[np.number])
X = data.drop("target", axis=1).values
y = data["target"].values.ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ======================================================
# 3. MLP Model Builder
# ======================================================
def build_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return model

# ======================================================
# 4. Model Functions
# ======================================================
def mlp_predict(X_train, y_train, X_test):
    model = build_mlp(X_train.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    return model.predict(X_test).ravel()

def mlp_bagging_predict(X_train, y_train, X_test, n_models=3):
    preds = []
    for seed in range(n_models):
        tf.random.set_seed(seed)
        model = build_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        preds.append(model.predict(X_test).ravel())
    return np.mean(preds, axis=0)

def mlp_rf_soft_voting(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = rf.predict_proba(X_test)[:, 1]
    mlp_score = mlp_bagging_predict(X_train, y_train, X_test, n_models=3)
    return 0.5 * rf_score + 0.5 * mlp_score

def mlp_xgb(X_train, y_train, X_test):
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,eval_metric="logloss", scale_pos_weight=5)
    xgb.fit(X_train, y_train)
    xgb_score = xgb.predict_proba(X_test)[:, 1]
    mlp_score = mlp_bagging_predict(X_train, y_train, X_test, n_models=3)
    return 0.5 *  xgb_score + 0.5 *  mlp_score

def stacking_ensemble(X_train, y_train, X_test, n_folds=3):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(X_train))
    oof_mlp = np.zeros(len(X_train))

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]

        mlp = build_mlp(X_tr.shape[1])
        mlp.fit(X_tr, y_tr, epochs=15, batch_size=32, verbose=0)
        oof_mlp[val_idx] = mlp.predict(X_val).ravel()

    meta = LogisticRegression()
    meta.fit(np.vstack([oof_rf, oof_mlp]).T, y_train)

    rf_test = RandomForestClassifier(
        n_estimators=100, random_state=42
    ).fit(X_train, y_train).predict_proba(X_test)[:, 1]

    mlp_test = build_mlp(X_train.shape[1])
    mlp_test.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    mlp_test = mlp_test.predict(X_test).ravel()

    return meta.predict_proba(np.vstack([rf_test, mlp_test]).T)[:, 1]

# ======================================================
# 5. Models dictionary
# ======================================================
models = {
    "MLP": mlp_predict,
    "Bagging": lambda Xtr, ytr, Xte: mlp_bagging_predict(Xtr, ytr, Xte, 3),
    "RF_Voting": mlp_rf_soft_voting,
    "XGBoost": mlp_xgb,
    "Stacking": stacking_ensemble
}

# ======================================================
# 6. K-run Evaluation
# ======================================================
results_k = {
    model: {m: [] for m in metrics}
    for model in models
}

for k in range(K):
    print(f"Run {k+1}/{K}")
    centralized = {m: {met: [] for met in metrics} for m in models}

    for name, func in models.items():
        start = time.time()
        y_score_all = func(X_train, y_train, X_test)
        latency = time.time() - start
        y_pred_all = (y_score_all > 0.5).astype(int)

        for r in ratios:
            n = int(len(X_test) * r)
            if n < 10:
                continue

            y_t = y_test[:n]
            y_p = y_pred_all[:n]
            y_s = y_score_all[:n]

            centralized[name]["accuracy"].append(accuracy_score(y_t, y_p))
            centralized[name]["precision"].append(
                precision_score(y_t, y_p, zero_division=0))
            centralized[name]["recall"].append(
                recall_score(y_t, y_p, zero_division=0))
            centralized[name]["f1"].append(
                f1_score(y_t, y_p, zero_division=0))
            centralized[name]["auc"].append(
                roc_auc_score(y_t, y_s) if len(np.unique(y_t)) > 1 else 0)
            centralized[name]["latency"].append(latency)

    for name in models:
        for m in metrics:
            results_k[name][m].append(np.mean(centralized[name][m]))

# ======================================================
# 7. Line Plot + Error Bar
# ======================================================
def plot_with_errorbar(results, metric):
    plt.figure(figsize=(8,5))
    for name in results:
        values = np.array(results[name][metric])
        values = np.sort(values)
        mean = np.mean(values)
        std = np.std(values)

        plt.errorbar(
            range(1, len(values)+1),
            values,
            yerr=std,
            marker='o',
            capsize=4,
            label=name
        )

    plt.xlabel("Iterations")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} over {K} Runs in Federals(±STD)")
    plt.grid(True)

    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout(rect=[0, 0, 0.99, 1])
    plt.show()

# ======================================================
# 8. Boxplot
# ======================================================
def plot_boxplot(results, metric):
    plt.figure(figsize=(7,5))
    plt.boxplot(
        [results[name][metric] for name in results],
        labels=results.keys(),
        showmeans=True
    )
    plt.ylabel(metric.upper())
    plt.title(f"Boxplot of Mean {metric.upper()} over {K} Runs in Federals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ======================================================
# 9. Plot All Metrics
# ======================================================
for m in metrics:
    plot_with_errorbar(results_k, m)
    plot_boxplot(results_k, m)

# ======================================================
# 10. Final Results
# ======================================================
print("\n=== Final Mean Metrics over K Runs ===")
for name in results_k:
    print(f"\n{name}")
    for m in metrics:
        print(f"{m}: {np.mean(results_k[name][m]):.4f}")
