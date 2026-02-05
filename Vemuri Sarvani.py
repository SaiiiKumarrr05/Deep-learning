# DL_final.py
# Approach: Human-in-the-loop (HITL) with calibrated probabilities, uncertainty-based routing,
# and append-only audit logs for accountability and reliability.

# Env: Python 3.10+ recommended; works on 3.8+. scikit-learn >=1.1, pandas, numpy.
# If your sklearn <1.2, the code auto-falls back for OneHotEncoder sparse argument.

import os, json, time, uuid, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

# ================== CONFIG ==================
csv_path = "/Users/saadhanagroup/Downloads/diabetes_dataset.csv"  # set to your actual Kaggle CSV filename
model_version = "v1.1"
t_high = 0.7     # auto-positive threshold if confident
t_low  = 0.3     # auto-negative threshold if confident
u_max  = 0.05    # uncertainty cap for auto-decisions (std of ensemble probs)
n_ens  = 5       # ensemble size for uncertainty
random_seed = 42

# ================== 1) LOAD ==================
df = pd.read_csv(csv_path)

# ================== 2) TARGET SELECTION (auto-detect) ==================
possible_targets = ["diagnosed_diabetes", "diabetes_stage", "Diabetes_binary"]
present_targets = [t for t in possible_targets if t in df.columns]
assert len(present_targets) > 0, f"No expected target found. Columns: {df.columns.tolist()}"
target_col = present_targets[0]

# If 'diabetes_stage' is multiclass and binary is desired, binarize here (adjust mapping to your labels).
if target_col == "diabetes_stage":
    stage = df[target_col].astype(str).str.lower().str.strip()
    y = (stage == "diabetes").astype(int).values  # mark 'diabetes' as 1, others (healthy/prediabetes) as 0
    X = df.drop(columns=[target_col])
else:
    y = df[target_col].values
    X = df.drop(columns=[target_col])

# ================== 3) FEATURE TYPES ==================
numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in numeric_cols]

print(f"Detected target: {target_col}")
uniq, cnts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(uniq.tolist(), cnts.tolist())))

# ================== 4) SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=random_seed, stratify=y
)

# ================== 5) PREPROCESS (sparse-friendly, no deprecation warnings) ==================
# Handle OneHotEncoder sparse argument across sklearn versions
def make_ohe_kwargs():
    ver_major, ver_minor = map(int, sklearn.__version__.split(".")[:2])
    if (ver_major, ver_minor) >= (1, 2):
        return {"handle_unknown": "ignore", "sparse_output": True}
    else:
        return {"handle_unknown": "ignore", "sparse": True}

ohe_kwargs = make_ohe_kwargs()

preprocess = ColumnTransformer(
    transformers=[
        ("num", MaxAbsScaler(), numeric_cols),                      # sparse-friendly scaling
        ("cat", OneHotEncoder(**ohe_kwargs), cat_cols)              # sparse OHE to avoid warnings/memory blowup
    ],
    remainder="drop"
)

# ================== 6) IMBALANCE HANDLING ==================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

# ================== 7) BASE MODELS ==================
lr = LogisticRegression(solver="liblinear", class_weight=class_weight_dict, max_iter=300, random_state=random_seed)
gb = GradientBoostingClassifier(random_state=random_seed)

lr_pipe = Pipeline([("prep", preprocess), ("clf", lr)])
gb_pipe = Pipeline([("prep", preprocess), ("clf", gb)])

# ================== 8) CALIBRATION (post-hoc sigmoid) ==================
lr_cal = CalibratedClassifierCV(lr_pipe, method="sigmoid", cv=3)
gb_cal = CalibratedClassifierCV(gb_pipe, method="sigmoid", cv=3)

lr_cal.fit(X_train, y_train)
gb_cal.fit(X_train, y_train)

def evaluate(model, X, y, name):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    brier = brier_score_loss(y, proba)
    return {"model": name, "AUC": auc, "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1, "Brier": brier}

print(evaluate(lr_cal, X_test, y_test, "LR+Cal"))
print(evaluate(gb_cal, X_test, y_test, "GB+Cal"))

# ================== 9) LIGHTWEIGHT UNCERTAINTY (ensembled calibrators) ==================
def ensemble_calibrated_predictions(base_pipe, X_train, y_train, X_eval, n=5, seed=0):
    # Refit calibrators multiple times to obtain variability; a proxy for epistemic uncertainty.
    probas = []
    rng = np.random.RandomState(seed)
    for k in range(n):
        cal = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)
        probas.append(cal.predict_proba(X_eval)[:, 1])
    P = np.vstack(probas)
    return P.mean(axis=0), P.std(axis=0)

lr_mean, lr_std = ensemble_calibrated_predictions(lr_pipe, X_train, y_train, X_test, n=n_ens, seed=7)
gb_mean, gb_std = ensemble_calibrated_predictions(gb_pipe, X_train, y_train, X_test, n=n_ens, seed=13)

# ================== 10) HITL ROUTING ==================
# 1 => auto-positive, 0 => auto-negative, -1 => route to human review
def hitl_route(mean_p, std_p, t_low, t_high, u_max):
    return np.where(
        (mean_p >= t_high) & (std_p <= u_max), 1,
        np.where((mean_p < t_low) & (std_p <= u_max), 0, -1)
    )

lr_actions = hitl_route(lr_mean, lr_std, t_low, t_high, u_max)
gb_actions = hitl_route(gb_mean, gb_std, t_low, t_high, u_max)

def hitl_summary(actions, y_true, label):
    total = len(actions)
    to_human = int(np.sum(actions == -1))
    auto = total - to_human
    print(f"{label} | total={total}, auto={auto}, human_review={to_human} ({to_human/total:.1%})")

hitl_summary(lr_actions, y_test, "LR+Cal Ensemble")
hitl_summary(gb_actions, y_test, "GB+Cal Ensemble")

# ================== 11) APPEND-ONLY AUDIT LOGS ==================
os.makedirs("audit_logs", exist_ok=True)
run_id = str(uuid.uuid4())
timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# Persist environment header for accountability
run_header = {
    "run_id": run_id,
    "timestamp_utc": timestamp,
    "env": {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__
    },
    "model_version": model_version,
    "thresholds": {"t_low": t_low, "t_high": t_high, "u_max": u_max},
    "target_col": target_col
}
with open("audit_logs/run_header.json", "w", encoding="utf-8") as f:
    json.dump(run_header, f, indent=2)

def write_audit(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

for model_name, mean_p, std_p, actions in [
    ("LR+Cal_Ens", lr_mean, lr_std, lr_actions),
    ("GB+Cal_Ens", gb_mean, gb_std, gb_actions),
]:
    fname = f"audit_logs/{model_name}.jsonl"
    for i in range(len(X_test)):
        record = {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "model_name": model_name,
            "model_version": model_version,
            "thresholds": {"t_low": t_low, "t_high": t_high, "u_max": u_max},
            "sample_index": int(i),
            "mean_proba": float(mean_p[i]),
            "uncertainty_std": float(std_p[i]),
            "auto_decision": int(actions[i]),  # 1/0 or -1 for human
            "ground_truth": int(y_test[i])
        }
        write_audit(fname, record)

print(f"Audit logs written to audit_logs/ (run_id={run_id})")
