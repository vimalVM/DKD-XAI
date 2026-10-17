"""
dkd_model_pipeline.py
=====================
Explainable AI for Early Prediction of Diabetic Kidney Disease (DKD)

Data   : dkd_nhanes_final.csv  (NHANES 2013-2014, 2015-2016, 2017-2020, 2021-2023)
Tasks  : 1) dkd_any   -> any kidney damage (UACR >= 30 or eGFR < 60)   vs none
         2) early_dkd -> early DKD (UACR 30-300, eGFR >= 60)           vs none
Inputs : only non-kidney variables (age, sex, HbA1c, BMI, waist, BP, lipids,
         diabetes treatment, duration). Kidney labs are NEVER used as inputs.

Models : Logistic Regression, Random Forest, XGBoost   (machine learning)
         PyTorch MLP                                    (deep learning)

Steps  : 80/20 stratified split -> tuned with 5-fold CV on the training part
         -> hold-out test metrics (+ bootstrap CI)
         -> temporal validation (train all earlier cycles, test newest cycle only)
         -> explainability: SHAP (XGBoost + MLP), permutation importance,
            logistic-regression odds ratios, explanation stability checks.

RUN
    pip install pandas numpy scikit-learn xgboost shap torch matplotlib scipy joblib
    python dkd_model_pipeline.py --csv dkd_nhanes_final.csv --out_dir results
    python dkd_model_pipeline.py --quick --skip_mlp --skip_mlp_shap --skip_temporal_validation
Optional flags: --tasks dkd_any early_dkd   --use_race   --skip_mlp_shap   --skip_mlp
                 --skip_temporal_validation
"""
import argparse
import itertools
import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.set_num_threads(1)

FEATURES = ["age", "female", "hba1c", "bmi", "waist_cm", "sbp", "dbp", "hdl",
            "total_chol", "dm_diagnosed", "insulin_use", "diabetes_pills",
            "dm_duration_years"]
NICE = {"age": "Age", "female": "Female sex", "hba1c": "HbA1c", "bmi": "BMI",
        "waist_cm": "Waist", "sbp": "Systolic BP", "dbp": "Diastolic BP",
        "hdl": "HDL", "total_chol": "Total cholesterol",
        "dm_diagnosed": "Diagnosed diabetes", "insulin_use": "Insulin use",
        "diabetes_pills": "Diabetes pills", "dm_duration_years": "Diabetes duration"}


# =============================================================================
# 1. DEEP LEARNING MODEL  (PyTorch MLP wrapped as a scikit-learn classifier)
# =============================================================================
class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Feed-forward network: [Linear -> BatchNorm -> ReLU -> Dropout] x N -> 1 logit.
    Class-weighted BCE loss, AdamW, early stopping on a 15 % internal validation split."""

    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-3,
                 epochs=200, batch_size=64, patience=15, random_state=SEED):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _build(self, n_in):
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, stratify=y,
                                              random_state=self.random_state)
        self.model_ = self._build(X.shape[1])
        pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            order = rng.permutation(len(Xtr))
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                if len(idx) < 2:          # BatchNorm needs >= 2 samples
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model_(Xtr_t[idx]).squeeze(1), ytr_t[idx])
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val = loss_fn(self.model_(Xva_t).squeeze(1), yva_t).item()
            if val < best - 1e-4:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32))))
        p = p.squeeze(1).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =============================================================================
# 2. DATA
# =============================================================================
def get_xy(df, task, use_race):
    d = df.copy()
    if task == "dkd_any":
        y = d["dkd_any"]
    else:                                   # early_dkd: drop 'advanced' rows (NaN label)
        d = d[d["early_dkd"].notna()]
        y = d["early_dkd"]
    X = d[FEATURES].copy()
    if use_race:
        race = pd.get_dummies(d["race_ethnicity_code"].astype(int), prefix="race").astype(float)
        X = pd.concat([X, race], axis=1)
    return (X.reset_index(drop=True), y.astype(int).reset_index(drop=True),
            d["cycle"].reset_index(drop=True))


# =============================================================================
# 3. MODELS + HYPER-PARAMETER SEARCH SPACES
# =============================================================================
def make_models(y_train, quick):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    impute = lambda: SimpleImputer(strategy="median", add_indicator=True)
    logreg_iters = 3 if quick else 6
    forest_iters = 5 if quick else 20
    xgb_iters = 6 if quick else 25
    mlp_iters = 2 if quick else 10
    cv_folds = 3 if quick else 5
    return {
        "LogReg": (
            Pipeline([("imp", impute()), ("sc", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}, logreg_iters, cv_folds),
        "RandomForest": (
            Pipeline([("imp", impute()),
                      ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                     n_jobs=1, random_state=SEED))]),
            {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [3, 5, 8, None],
             "clf__min_samples_leaf": [3, 5, 10, 20], "clf__max_features": ["sqrt", 0.5]},
            forest_iters, cv_folds),
        "XGBoost": (
            Pipeline([("clf", xgb.XGBClassifier(tree_method="hist", eval_metric="logloss",
                                                scale_pos_weight=spw, n_jobs=1,
                                                random_state=SEED))]),
            {"clf__n_estimators": [100, 200, 300, 500], "clf__max_depth": [2, 3, 4, 5],
             "clf__learning_rate": [0.01, 0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5, 10],
             "clf__reg_lambda": [1, 5, 10]}, xgb_iters, cv_folds),
        "MLP (deep learning)": (
            Pipeline([("imp", impute()), ("sc", StandardScaler()),
                      ("clf", TorchMLPClassifier(epochs=60 if quick else 200))]),
            {"clf__hidden": [(32,), (64,), (64, 32), (128, 64), (64, 32, 16)],
             "clf__dropout": [0.1, 0.2, 0.3, 0.5], "clf__lr": [3e-4, 1e-3, 3e-3],
             "clf__weight_decay": [1e-4, 1e-3, 1e-2]}, mlp_iters, 3),
    }


# =============================================================================
# 4. METRICS
# =============================================================================
def youden_threshold(y, p):
    """Choose Youden's J threshold, excluding sklearn's leading infinite threshold."""
    fpr, tpr, thr = roc_curve(y, p)
    finite = np.isfinite(thr)
    if not finite.any():
        return 0.5
    j = tpr[finite] - fpr[finite]
    return float(thr[finite][int(np.argmax(j))])


def _cycle_start_year(value):
    """Return the first four-digit year from a cycle label for chronological ordering."""
    import re
    match = re.search(r"(?:19|20)\d{2}", str(value))
    return int(match.group(0)) if match else -1


def temporal_holdout_masks(cycle):
    """Use the newest observed cycle as an untouched temporal test cohort."""
    values = cycle.astype(str).reset_index(drop=True)
    observed = values.unique().tolist()
    if len(observed) < 2:
        return None, None, None
    newest = max(observed, key=_cycle_start_year)
    train_mask = (values != newest).to_numpy()
    test_mask = (values == newest).to_numpy()
    if not train_mask.any() or not test_mask.any():
        return None, None, newest
    return train_mask, test_mask, newest


def run_temporal_validation(X, y, cycle, model_name, args):
    """Tune using only pre-newest cycles, then evaluate on the newest cycle.

    This keeps every row in the temporal test cycle out of both tuning and fitting.
    """
    train_mask, test_mask, newest = temporal_holdout_masks(cycle)
    if train_mask is None:
        return None

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    class_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    if len(class_counts) < 2 or len(test_counts) < 2:
        print(f"  Temporal validation skipped for {model_name}: one class absent in train/test.")
        return None

    temporal_models = make_models(y_train, args.quick)
    pipe, space, n_iter, requested_folds = temporal_models[model_name]
    min_class = int(class_counts.min())
    n_splits = min(requested_folds, min_class)
    if n_splits < 2:
        print(f"  Temporal validation skipped for {model_name}: too few minority-class train rows.")
        return None

    search = RandomizedSearchCV(
        pipe, space,
        n_iter=min(n_iter, int(np.prod([len(v) for v in space.values()]))),
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    threshold_folds = min(5, int(class_counts.min()))
    oof = cross_val_predict(
        clone(best), X_train, y_train,
        cv=StratifiedKFold(n_splits=threshold_folds, shuffle=True, random_state=SEED + 1),
        method="predict_proba",
    )[:, 1]
    threshold = youden_threshold(y_train, oof)
    probabilities = best.predict_proba(X_test)[:, 1]
    ci_low, ci_high = bootstrap_auc_ci(
        y_test, probabilities, n=300 if args.quick else 1000, seed=SEED + 1
    )
    metrics = threshold_metrics(y_test, probabilities, threshold)
    return {
        "model": model_name,
        "train_cycles": ";".join(sorted(cycle[train_mask].astype(str).unique(), key=_cycle_start_year)),
        "test_cycle": newest,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "cv_auroc_train": float(search.best_score_),
        "auroc": float(roc_auc_score(y_test, probabilities)),
        "auroc_ci_low": float(ci_low),
        "auroc_ci_high": float(ci_high),
        "auprc": float(average_precision_score(y_test, probabilities)),
        "brier": float(brier_score_loss(y_test, probabilities)),
        "threshold": float(threshold),
        **metrics,
    }


def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "accuracy": (tp + tn) / len(y), "f1": f1_score(y, pred)}


def bootstrap_auc_ci(y, p, n=1000, seed=SEED):
    rng, y, p, vals = np.random.RandomState(seed), np.asarray(y), np.asarray(p), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


# =============================================================================
# 5. EXPLAINABILITY HELPERS
# =============================================================================
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def xgb_shap(pipe, X):
    """Return positive-class SHAP values and a matching base value across SHAP versions."""
    model = pipe.named_steps["clf"]
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    expected = expl.expected_value
    if isinstance(sv, list):
        class_index = 1 if len(sv) > 1 else 0
        sv = sv[class_index]
        expected_array = np.ravel(expected)
        expected = expected_array[class_index] if expected_array.size > class_index else expected_array[0]
    else:
        sv = np.asarray(sv)
        expected_array = np.ravel(expected)
        if sv.ndim == 3:
            class_index = sv.shape[-1] - 1
            sv = sv[:, :, class_index]
            expected = expected_array[class_index] if expected_array.size > class_index else expected_array[0]
        else:
            expected = expected_array[-1] if expected_array.size > 1 else expected_array[0]
    return np.asarray(sv), float(expected)


def top_k(imp, cols, k=5):
    return set(np.array(cols)[np.argsort(-imp)[:k]])


def explanation_stability(best_xgb, X_tr, y_tr, X_te, n_boot, out_dir):
    """Refit XGBoost on bootstrap resamples; how stable is the SHAP feature ranking?"""
    rng, imps = np.random.RandomState(SEED), []
    for _ in range(n_boot):
        idx = rng.randint(0, len(X_tr), len(X_tr))
        if y_tr.iloc[idx].nunique() < 2:
            continue
        m = clone(best_xgb).fit(X_tr.iloc[idx], y_tr.iloc[idx])
        imps.append(np.abs(xgb_shap(m, X_te)[0]).mean(axis=0))
    imps = np.array(imps)
    rho = [spearmanr(imps[i], imps[j])[0] for i, j in itertools.combinations(range(len(imps)), 2)]
    tops = [top_k(v, X_te.columns) for v in imps]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
    res = {"bootstrap_fits": len(imps), "mean_spearman_rank_corr": float(np.mean(rho)),
           "mean_top5_jaccard": float(np.mean(jac))}
    mean, sd = imps.mean(axis=0), imps.std(axis=0)
    order = np.argsort(mean)
    plt.figure(figsize=(7, 5))
    plt.barh([NICE.get(c, c) for c in X_te.columns[order]], mean[order], xerr=sd[order], color="#3b6fb6")
    plt.xlabel("mean |SHAP| (mean +/- SD over bootstrap refits)")
    plt.title("XGBoost explanation stability")
    savefig(os.path.join(out_dir, "shap_bootstrap_stability.png"))
    return res


# =============================================================================
# 6. MAIN EXPERIMENT FOR ONE TASK
# =============================================================================
def run_task(df, task, args):
    out = os.path.join(args.out_dir, task)
    os.makedirs(out, exist_ok=True)
    X, y, cycle = get_xy(df, task, args.use_race)
    print(f"\n{'=' * 70}\nTASK: {task}   n={len(y)}   positives={int(y.sum())} ({y.mean():.1%})\n{'=' * 70}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv5 = StratifiedKFold(3 if args.quick else 5, shuffle=True, random_state=SEED)
    models = make_models(y_tr, args.quick)
    if args.skip_mlp:
        models.pop("MLP (deep learning)", None)

    rows, probs, fitted, thresholds, temporal_rows = [], {}, {}, {}, []
    for name, (pipe, space, n_iter, folds) in models.items():
        size = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(
            pipe, space, n_iter=min(n_iter, size), scoring="roc_auc",
            cv=StratifiedKFold(folds, shuffle=True, random_state=SEED), random_state=SEED,
            n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        fitted[name] = best

        # threshold chosen on TRAIN out-of-fold predictions (no test leakage)
        oof = cross_val_predict(clone(best), X_tr, y_tr, cv=cv5, method="predict_proba")[:, 1]
        thr = youden_threshold(y_tr, oof)
        thresholds[name] = thr

        p = best.predict_proba(X_te)[:, 1]
        probs[name] = p
        lo, hi = bootstrap_auc_ci(y_te, p)
        row = {"model": name, "cv_auroc_train": search.best_score_,
               "test_auroc": roc_auc_score(y_te, p), "auroc_ci_low": lo, "auroc_ci_high": hi,
               "test_auprc": average_precision_score(y_te, p),
               "brier": brier_score_loss(y_te, p), "threshold": thr,
               **threshold_metrics(y_te, p, thr), "best_params": str(search.best_params_)}
        rows.append(row)
        print(f"  {name:22s} CV AUROC {search.best_score_:.3f} | test AUROC {row['test_auroc']:.3f} "
              f"[{lo:.3f}-{hi:.3f}] | AUPRC {row['test_auprc']:.3f}")

        # Independent temporal validation: all rows in the newest cycle are untouched
        # by hyperparameter selection and fitting for this validation result.
        if not args.skip_temporal_validation:
            temporal_result = run_temporal_validation(X, y, cycle, name, args)
            if temporal_result is not None:
                temporal_rows.append(temporal_result)

    res = pd.DataFrame(rows).sort_values("cv_auroc_train", ascending=False)
    res.to_csv(os.path.join(out, "model_comparison_holdout.csv"), index=False)
    temporal_df = pd.DataFrame(temporal_rows)
    temporal_df.to_csv(os.path.join(out, "model_comparison_temporal.csv"), index=False)
    print("\nTemporal validation (tune/train on all earlier cycles -> test newest cycle):")
    print(temporal_df.round(3).to_string(index=False) if not temporal_df.empty else "No temporal results produced.")

    # ---- ROC / PR / calibration plots ---------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y_te, p):.3f})")
        fp, mp = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
        ax[2].plot(mp, fp, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(y_te.mean(), color="k", ls="--", lw=0.8)
    ax[1].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision")
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[2].set(title="Calibration (test)", xlabel="Predicted risk", ylabel="Observed risk")
    for a in ax:
        a.legend(fontsize=8)
    savefig(os.path.join(out, "roc_pr_calibration.png"))

    # ---- select best model by CROSS-VALIDATION score (not by test) ----------
    selected = res.sort_values("cv_auroc_train", ascending=False).iloc[0]["model"]
    print(f"\nSelected model (best CV AUROC): {selected}")
    bundle = {"model": fitted[selected], "features": list(X.columns),
              "threshold": thresholds[selected], "task": task, "name": selected}
    joblib.dump(bundle, os.path.join(out, "best_model.joblib"))

    # ==================== EXPLAINABILITY =====================================
    # (a) Logistic-regression odds ratios (per +1 SD) - simplest transparent baseline
    lr = fitted["LogReg"]
    names = lr.named_steps["imp"].get_feature_names_out(list(X.columns))
    orr = pd.DataFrame({"feature": names, "odds_ratio_per_SD": np.exp(lr.named_steps["clf"].coef_[0])})
    orr.sort_values("odds_ratio_per_SD", ascending=False).to_csv(
        os.path.join(out, "logreg_odds_ratios.csv"), index=False)

    # (b) Permutation importance on the selected model (model-agnostic)
    pi = permutation_importance(fitted[selected], X_te, y_te, scoring="roc_auc",
                                n_repeats=20, random_state=SEED)
    order = np.argsort(pi.importances_mean)
    plt.figure(figsize=(7, 5))
    plt.barh([NICE.get(c, c) for c in X.columns[order]], pi.importances_mean[order],
             xerr=pi.importances_std[order], color="#c46a2d")
    plt.xlabel("Drop in test AUROC when feature is shuffled")
    plt.title(f"Permutation importance - {selected}")
    savefig(os.path.join(out, "permutation_importance.png"))

    # (c) SHAP for XGBoost (exact TreeSHAP)
    sv, base = xgb_shap(fitted["XGBoost"], X_te)
    shap_df = pd.DataFrame(sv, columns=X.columns)
    shap_df.to_csv(os.path.join(out, "shap_values_xgboost_test.csv"), index=False)
    Xn = X_te.rename(columns=NICE)
    shap.summary_plot(sv, Xn, show=False, plot_size=(8, 5))
    savefig(os.path.join(out, "shap_summary_beeswarm.png"))
    shap.summary_plot(sv, Xn, plot_type="bar", show=False, plot_size=(7, 5))
    savefig(os.path.join(out, "shap_summary_bar.png"))
    imp_xgb = np.abs(sv).mean(axis=0)
    for f in np.array(X.columns)[np.argsort(-imp_xgb)[:3]]:
        try:
            shap.dependence_plot(NICE.get(f, f), sv, Xn, show=False)
            savefig(os.path.join(out, f"shap_dependence_{f}.png"))
        except Exception as e:
            print("  dependence plot skipped for", f, e)
    p_x = fitted["XGBoost"].predict_proba(X_te)[:, 1]
    for tag, i in {"highest_risk": int(np.argmax(p_x)), "lowest_risk": int(np.argmin(p_x)),
                   "median_risk": int(np.argsort(p_x)[len(p_x) // 2])}.items():
        shap.plots.waterfall(shap.Explanation(values=sv[i], base_values=base, data=Xn.iloc[i].values,
                                              feature_names=list(Xn.columns)), show=False)
        savefig(os.path.join(out, f"shap_patient_{tag}.png"))

    # (d) SHAP for the deep-learning MLP (model-agnostic KernelSHAP on a subset)
    stability = {}
    if not args.skip_mlp_shap and not args.skip_mlp:
        mlp = fitted["MLP (deep learning)"]
        f = lambda A: mlp.predict_proba(pd.DataFrame(A, columns=X.columns))[:, 1]
        bg = X_tr.sample(min(50, len(X_tr)), random_state=SEED)
        sub = X_te.sample(min(60 if args.quick else 150, len(X_te)), random_state=SEED)
        sv_mlp = shap.KernelExplainer(f, bg).shap_values(sub, nsamples=100 if args.quick else 200, silent=True)
        sv_mlp = np.asarray(sv_mlp[1] if isinstance(sv_mlp, list) else sv_mlp)
        imp_mlp = np.abs(sv_mlp).mean(axis=0)
        imp_x_sub = np.abs(xgb_shap(fitted["XGBoost"], sub)[0]).mean(axis=0)
        stability["xgb_vs_mlp_spearman"] = float(spearmanr(imp_x_sub, imp_mlp)[0])
        stability["xgb_vs_mlp_top5_overlap"] = len(top_k(imp_x_sub, X.columns) & top_k(imp_mlp, X.columns))
        w = 0.4
        o = np.argsort(imp_x_sub / imp_x_sub.sum())
        yy = np.arange(len(o))
        plt.figure(figsize=(8, 5.5))
        plt.barh(yy - w / 2, (imp_x_sub / imp_x_sub.sum())[o], w, label="XGBoost (TreeSHAP)", color="#3b6fb6")
        plt.barh(yy + w / 2, (imp_mlp / imp_mlp.sum())[o], w, label="MLP (KernelSHAP)", color="#c46a2d")
        plt.yticks(yy, [NICE.get(c, c) for c in X.columns[o]])
        plt.xlabel("Share of total mean |SHAP|")
        plt.title("Do ML and DL models rely on the same features?")
        plt.legend()
        savefig(os.path.join(out, "shap_xgboost_vs_mlp.png"))

    # (e) Bootstrap stability of the XGBoost explanation
    stability.update(explanation_stability(fitted["XGBoost"], X_tr, y_tr, X_te,
                                           8 if args.quick else 25, out))
    stability["top5_features_xgboost_shap"] = [NICE.get(c, c) for c in
                                               np.array(X.columns)[np.argsort(-imp_xgb)[:5]]]
    json.dump(stability, open(os.path.join(out, "explanation_stability.json"), "w"), indent=2)
    print("\nExplanation stability:", json.dumps(stability, indent=2))

    # ---- example prediction for a single new patient --------------------------
    example = X_te.iloc[0].to_dict()
    print("\nExample patient:", {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in example.items()})
    print("Predicted risk:", predict_patient(os.path.join(out, "best_model.joblib"), example))
    return res


# =============================================================================
# 7. USE THE SAVED MODEL ON A NEW PATIENT
# =============================================================================
def predict_patient(model_path, patient: dict):
    """patient: dict of feature -> value (leave a feature out or use None if unknown)."""
    import __main__
    if not hasattr(__main__, "TorchMLPClassifier"):       # needed to unpickle the MLP
        __main__.TorchMLPClassifier = TorchMLPClassifier
    b = joblib.load(model_path)
    row = pd.DataFrame([{f: patient.get(f, np.nan) for f in b["features"]}], dtype=float)
    risk = float(b["model"].predict_proba(row)[0, 1])
    return {"model": b["name"], "task": b["task"], "risk_probability": round(risk, 3),
            "high_risk": bool(risk >= b["threshold"]), "threshold": round(b["threshold"], 3)}


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dkd_nhanes_final.csv")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--tasks", nargs="+", default=["dkd_any", "early_dkd"],
                    choices=["dkd_any", "early_dkd"])
    ap.add_argument("--use_race", action="store_true", help="add race/ethnicity dummies as inputs")
    ap.add_argument("--quick", action="store_true", help="small search + fewer bootstraps (smoke test)")
    ap.add_argument("--skip_mlp_shap", action="store_true", help="skip slow KernelSHAP for the MLP")
    ap.add_argument("--skip_mlp", action="store_true",
                    help="skip deep-learning model training (use for smoke tests, not final comparisons)")
    ap.add_argument("--skip_temporal_validation", action="store_true",
                    help="skip the independent chronological holdout (use for smoke tests only)")
    args = ap.parse_args()

    data = pd.read_csv(args.csv)
    summary = {}
    for t in args.tasks:
        summary[t] = run_task(data, t, args)
    print("\n\nFINAL SUMMARY (hold-out test set)")
    for t, r in summary.items():
        print(f"\n{t}")
        print(r[["model", "cv_auroc_train", "test_auroc", "auroc_ci_low", "auroc_ci_high",
                 "test_auprc", "sensitivity", "specificity"]].round(3).to_string(index=False))
    print(f"\nAll tables, figures and saved models are in: {args.out_dir}/")
