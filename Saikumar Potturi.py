# DL_final_leakage_reduced_with_plots.py
# Accountability-first diabetes triage: calibrated probabilities, uncertainty-based routing (HITL),
# leakage-aware features, metrics and plots, and append-only audit logs.

import os, json, time, uuid, platform
import numpy as np
import pandas as pd
import sklearn

# Ensure plotting works in VS Code and also saves when headless
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.class_weight import compute_class_weight

# ================== CONFIG ==================
csv_path = "/Users/saadhanagroup/Downloads/diabetes_dataset.csv"  # set to your CSV
model_version = "v1.4-leakage-reduced-plots"
t_high = 0.7
t_low  = 0.3
u_max  = 0.05
n_ens  = 5
random_seed = 42
outdir = "reports"
os.makedirs(outdir, exist_ok=True)

# ================== 1) LOAD ==================
df = pd.read_csv(csv_path)

# ================== 2) TARGET SELECTION (auto) ==================
possible_targets = ["diagnosed_diabetes", "diabetes_stage", "Diabetes_binary"]
present_targets = [t for t in possible_targets if t in df.columns]
assert len(present_targets) > 0, f"No expected target found. Columns: {df.columns.tolist()}"
target_col = present_targets[0]

if target_col == "diabetes_stage":
    stage = df[target_col].astype(str).str.lower().str.strip()
    y = (stage == "diabetes").astype(int).values
    X_full = df.drop(columns=[target_col])
else:
    y = df[target_col].values
    X_full = df.drop(columns=[target_col])

# ================== 3) LEAKAGE-AWARE FEATURE FILTERING ==================
leak_like = {
    "diagnosed_diabetes", "diabetes_stage", "diabetes_risk_score",
    "hba1c", "glucose_fasting", "glucose_postprandial", "insulin_level",
    "waist_to_hip_ratio"
}
drop_cols = [c for c in X_full.columns if c.lower() in leak_like]
X = X_full.drop(columns=drop_cols, errors="ignore")

print(f"Detected target: {target_col}")
print(f"Dropping leakage-prone columns: {drop_cols}")

uniq, cnts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(uniq.tolist(), cnts.tolist())))
print("Remaining features:", len(X.columns))

# ================== 4) SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=random_seed, stratify=y
)

# ================== 5) PREPROCESS (sparse-friendly, no OHE warnings) ==================
def make_ohe_kwargs():
    ver_major, ver_minor = map(int, sklearn.__version__.split(".")[:2])
    if (ver_major, ver_minor) >= (1, 2):
        return {"handle_unknown": "ignore", "sparse_output": True}
    else:
        return {"handle_unknown": "ignore", "sparse": True}

ohe_kwargs = make_ohe_kwargs()

numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
cat_cols = [c for c in X_train.columns if c not in numeric_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", MaxAbsScaler(), numeric_cols),
        ("cat", OneHotEncoder(**ohe_kwargs), cat_cols)
    ],
    remainder="drop"
)

# ================== 6) IMBALANCE HANDLING ==================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

# ================== 7) BASE MODELS ==================
lr = LogisticRegression(solver="liblinear", class_weight=class_weight_dict, max_iter=400, random_state=random_seed)
gb = GradientBoostingClassifier(random_state=random_seed)

lr_pipe = Pipeline([("prep", preprocess), ("clf", lr)])
gb_pipe = Pipeline([("prep", preprocess), ("clf", gb)])

# ================== 8) CALIBRATION ==================
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
    return {"model": name, "AUC": auc, "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1, "Brier": brier, "proba": proba, "pred": pred}

print("Leakage-reduced metrics (nominal 0.5 threshold):")
m_lr = evaluate(lr_cal, X_test, y_test, "LR+Cal")
m_gb = evaluate(gb_cal, X_test, y_test, "GB+Cal")
print(m_lr)
print(m_gb)

# ================== 9) LIGHTWEIGHT UNCERTAINTY (ensembled calibrators) ==================
def ensemble_calibrated_predictions(base_pipe, X_train, y_train, X_eval, n=5, seed=0):
    probas = []
    for k in range(n):
        cal = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)
        probas.append(cal.predict_proba(X_eval)[:, 1])
    P = np.vstack(probas)
    return P.mean(axis=0), P.std(axis=0)

lr_mean, lr_std = ensemble_calibrated_predictions(lr_pipe, X_train, y_train, X_test, n=n_ens, seed=7)
gb_mean, gb_std = ensemble_calibrated_predictions(gb_pipe, X_train, y_train, X_test, n=n_ens, seed=13)

# ================== 10) FORMATTED METRICS ==================
for name, model in [("LR+Cal", lr_cal), ("GB+Cal", gb_cal)]:
    m = evaluate(model, X_test, y_test, name)
    print(f"{m['model']}: AUC={m['AUC']:.8f} | Accuracy={m['Accuracy']:.5f} | "
          f"Precision={m['Precision']:.12f} | Recall={m['Recall']:.12f} | "
          f"F1={m['F1']:.12f} | Brier={m['Brier']:.12f}")

# ================== PLOT HELPERS ==================
def plot_roc(y_true, proba_dict, save_path):
    plt.figure(figsize=(6,5))
    for label, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_pr(y_true, proba_dict, save_path):
    plt.figure(figsize=(6,5))
    for label, proba in proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        plt.plot(recall, precision, label=f"{label}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def expected_calibration_error(y_true, proba, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (proba >= lo) & (proba < hi)
        if np.any(idx):
            conf = proba[idx].mean()
            acc = y_true[idx].mean()
            ece += (idx.mean()) * abs(acc - conf)
    return ece

def plot_calibration(y_true, proba_dict, save_path):
    plt.figure(figsize=(6,5))
    for label, proba in proba_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=15, strategy="uniform")
        ece = expected_calibration_error(y_true, proba, n_bins=15)
        plt.plot(mean_pred, frac_pos, marker="o", label=f"{label} (ECE={ece:.3f})")
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_uncertainty(std_dict, save_path):
    plt.figure(figsize=(6,5))
    for label, stds in std_dict.items():
        sns.histplot(stds, bins=30, label=label, stat="density", kde=True, alpha=0.4)
    plt.xlabel("Prediction std (ensemble)")
    plt.ylabel("Density")
    plt.title("Uncertainty Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()

# ================== GENERATE PLOTS ==================
probas = {"LR+Cal": m_lr["proba"], "GB+Cal": m_gb["proba"]}
stds = {"LR ens std": lr_std, "GB ens std": gb_std}

plot_roc(y_test, probas, os.path.join(outdir, "roc.png"))
plot_pr(y_test, probas, os.path.join(outdir, "pr.png"))
plot_calibration(y_test, probas, os.path.join(outdir, "calibration.png"))
plot_uncertainty(stds, os.path.join(outdir, "uncertainty_hist.png"))

# Confusion matrices at 0.5 threshold
plot_confusion(y_test, (m_lr["proba"]>=0.5).astype(int),
               "Confusion Matrix (LR+Cal, p>=0.5)", os.path.join(outdir,"cm_lr_05.png"))
plot_confusion(y_test, (m_gb["proba"]>=0.5).astype(int),
               "Confusion Matrix (GB+Cal, p>=0.5)", os.path.join(outdir,"cm_gb_05.png"))

# HITL decisions and confusion matrices for auto-decided only
def hitl_decisions(mean_p, std_p, t_low, t_high, u_max):
    return np.where(
        (mean_p >= t_high) & (std_p <= u_max), 1,
        np.where((mean_p < t_low) & (std_p <= u_max), 0, -1)
    )

lr_act = hitl_decisions(lr_mean, lr_std, t_low, t_high, u_max)
gb_act = hitl_decisions(gb_mean, gb_std, t_low, t_high, u_max)

def auto_confusion(y_true, actions):
    mask = actions != -1
    return y_true[mask], actions[mask], mask.sum(), len(y_true)-mask.sum()

y_lr_auto_y, y_lr_auto_pred, n_auto_lr, n_human_lr = auto_confusion(y_test, lr_act)
y_gb_auto_y, y_gb_auto_pred, n_auto_gb, n_human_gb = auto_confusion(y_test, gb_act)

if n_auto_lr > 0:
    plot_confusion(y_lr_auto_y, y_lr_auto_pred,
                   f"Confusion (LR+Cal HITL auto-only) Auto={n_auto_lr}, Routed={n_human_lr}",
                   os.path.join(outdir,"cm_lr_hitl_auto.png"))
if n_auto_gb > 0:
    plot_confusion(y_gb_auto_y, y_gb_auto_pred,
                   f"Confusion (GB+Cal HITL auto-only) Auto={n_auto_gb}, Routed={n_human_gb}",
                   os.path.join(outdir,"cm_gb_hitl_auto.png"))

# ================== 11) APPEND-ONLY AUDIT LOGS ==================
os.makedirs("audit_logs", exist_ok=True)
run_id = str(uuid.uuid4())
timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

run_header = {
    "run_id": run_id,
    "timestamp_utc": timestamp,
    "env": {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__
    },
    "conda_python_path_hint": "/opt/anaconda3/envs/mlenv/bin/python",
    "model_version": model_version,
    "thresholds": {"t_low": t_low, "t_high": t_high, "u_max": u_max},
    "target_col": target_col,
    "dropped_columns": drop_cols,
    "reports": {
        "roc": os.path.join(outdir,"roc.png"),
        "pr": os.path.join(outdir,"pr.png"),
        "calibration": os.path.join(outdir,"calibration.png"),
        "uncertainty_hist": os.path.join(outdir,"uncertainty_hist.png"),
        "cm_lr_05": os.path.join(outdir,"cm_lr_05.png"),
        "cm_gb_05": os.path.join(outdir,"cm_gb_05.png"),
        "cm_lr_hitl_auto": os.path.join(outdir,"cm_lr_hitl_auto.png") if n_auto_lr>0 else None,
        "cm_gb_hitl_auto": os.path.join(outdir,"cm_gb_hitl_auto.png") if n_auto_gb>0 else None
    }
}
with open("audit_logs/run_header.json", "w", encoding="utf-8") as f:
    json.dump(run_header, f, indent=2)

# Optional: persist console lines you previously shared
reported_console = [
    "Detected target: diagnosed_diabetes",
    "Class distribution: {0: 40002, 1: 59998}",
    "{'model': 'LR+Cal', 'AUC': 0.99999775, 'Accuracy': 0.99905, 'Precision': 0.9992499374947912, 'Recall': 0.9991666666666666, 'F1': 0.9992083003458476, 'Brier': 0.0006473138053020934}",
    "{'model': 'GB+Cal', 'AUC': 0.9999986458333333, 'Accuracy': 0.9997, 'Precision': 1.0, 'Recall': 0.9995, 'F1': 0.9997499374843711, 'Brier': 0.00029136796719025546}"
]
with open("audit_logs/reported_console.json", "w", encoding="utf-8") as f:
    json.dump(reported_console, f, indent=2)

print(f"Saved plots to: {outdir}")
print(f"Audit header written to audit_logs/run_header.json (run_id={run_id})")
