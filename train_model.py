"""
CareerAI — Advanced Model Training Script
==========================================
Tackles class imbalance using 4 complementary strategies:

  Strategy A — Synthetic balanced data generation (manual SMOTE-style)
  Strategy B — class_weight='balanced' in RandomForest
  Strategy C — sample_weight (computed per-row) for GradientBoosting
  Strategy D — SMOTE (applied automatically if imbalanced-learn installed)
  Strategy E — Probability calibration via CalibratedClassifierCV
  Strategy F — Soft-voting ensemble (GB + RF averaged)

Install:
    pip install scikit-learn xgboost pandas numpy joblib
    pip install imbalanced-learn   # optional but recommended

Run:
    python train_model.py

Output:
    career_model.pkl      — drop into your Flask project root
    training_report.txt   — per-class accuracy breakdown
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# STEP 1 — CAREER PROFILES
# Each entry defines the realistic distribution of
# features for that career. The synthetic generator
# samples from these to produce balanced training data.
# ─────────────────────────────────────────────────────────

CAREER_PROFILES = {
    "Data Scientist": {
        "streams":       ["Science", "Engineering"],
        "interest_sets": ["AI,Data,Coding", "Data,Coding,Business",
                          "AI,Data,Finance", "AI,Coding,Research"],
        "degrees":       ["btech", "bsc", "mtech", "msc"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["analytical", "technical"],
        "goal":          ["high_salary", "passion", "research"],
        "tenth_range":   (75, 98), "twelfth_range": (72, 97), "grad_range": (70, 95),
    },
    "ML Engineer": {
        "streams":       ["Engineering", "Science"],
        "interest_sets": ["AI,Coding,Data", "AI,Coding,Research", "Coding,Data,Security"],
        "degrees":       ["btech", "mtech"],
        "coding":        ["advanced"],
        "work_type":     ["technical"],
        "goal":          ["high_salary", "passion"],
        "tenth_range":   (78, 99), "twelfth_range": (76, 98), "grad_range": (72, 96),
    },
    "Software Developer": {
        "streams":       ["Engineering", "Science", "Commerce"],
        "interest_sets": ["Coding,Data,Design", "Coding,Business,Data",
                          "Coding,AI", "Coding,Design"],
        "degrees":       ["btech", "bca", "bsc"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["technical", "creative"],
        "goal":          ["high_salary", "job_security"],
        "tenth_range":   (65, 98), "twelfth_range": (62, 97), "grad_range": (60, 95),
    },
    "Backend Developer": {
        "streams":       ["Engineering", "Science"],
        "interest_sets": ["Coding,Data,Security", "Coding,Data", "Coding,Business"],
        "degrees":       ["btech", "bca", "bsc"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["technical"],
        "goal":          ["high_salary", "job_security"],
        "tenth_range":   (65, 96), "twelfth_range": (62, 95), "grad_range": (60, 93),
    },
    "Frontend Developer": {
        "streams":       ["Engineering", "Science", "Arts"],
        "interest_sets": ["Coding,Design,Marketing", "Design,Coding",
                          "Coding,Design,Business"],
        "degrees":       ["btech", "bca", "bsc", "bdes"],
        "coding":        ["beginner", "intermediate", "advanced"],
        "work_type":     ["creative", "technical"],
        "goal":          ["passion", "high_salary"],
        "tenth_range":   (60, 95), "twelfth_range": (58, 93), "grad_range": (55, 92),
    },
    "Cloud Engineer": {
        "streams":       ["Engineering"],
        "interest_sets": ["Coding,Security,Data", "Coding,Data"],
        "degrees":       ["btech", "mtech"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["technical"],
        "goal":          ["high_salary", "job_security"],
        "tenth_range":   (70, 97), "twelfth_range": (68, 96), "grad_range": (65, 95),
    },
    "Cybersecurity Analyst": {
        "streams":       ["Engineering", "Science"],
        "interest_sets": ["Security,Coding,Data", "Security,Coding",
                          "Security,Data,Law"],
        "degrees":       ["btech", "bsc", "bca"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["analytical", "technical"],
        "goal":          ["high_salary", "passion"],
        "tenth_range":   (70, 97), "twelfth_range": (68, 96), "grad_range": (65, 94),
    },
    "DevOps Engineer": {
        "streams":       ["Engineering"],
        "interest_sets": ["Coding,Data,Security", "Coding,Data"],
        "degrees":       ["btech"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["technical"],
        "goal":          ["high_salary", "job_security"],
        "tenth_range":   (68, 96), "twelfth_range": (65, 95), "grad_range": (62, 93),
    },
    "Database Administrator": {
        "streams":       ["Engineering", "Science", "Commerce"],
        "interest_sets": ["Data,Coding,Business", "Data,Finance"],
        "degrees":       ["btech", "bca", "bsc", "bcom"],
        "coding":        ["intermediate"],
        "work_type":     ["analytical", "technical"],
        "goal":          ["job_security"],
        "tenth_range":   (60, 92), "twelfth_range": (58, 90), "grad_range": (55, 88),
    },
    "Network Engineer": {
        "streams":       ["Engineering"],
        "interest_sets": ["Security,Coding,Data", "Security,Data"],
        "degrees":       ["btech"],
        "coding":        ["beginner", "intermediate"],
        "work_type":     ["technical"],
        "goal":          ["job_security"],
        "tenth_range":   (60, 92), "twelfth_range": (58, 90), "grad_range": (55, 88),
    },
    "AI Researcher": {
        "streams":       ["Science", "Engineering"],
        "interest_sets": ["AI,Data,Research", "AI,Coding,Research"],
        "degrees":       ["mtech", "msc", "phd"],
        "coding":        ["advanced"],
        "work_type":     ["analytical", "research"],
        "goal":          ["research", "passion"],
        "tenth_range":   (85, 100), "twelfth_range": (83, 100), "grad_range": (80, 100),
    },
    "Blockchain Developer": {
        "streams":       ["Engineering", "Science"],
        "interest_sets": ["Coding,Finance,Data", "Coding,Finance"],
        "degrees":       ["btech", "bsc"],
        "coding":        ["advanced"],
        "work_type":     ["technical"],
        "goal":          ["high_salary", "passion"],
        "tenth_range":   (72, 98), "twelfth_range": (70, 97), "grad_range": (68, 96),
    },
    "Game Developer": {
        "streams":       ["Engineering", "Arts"],
        "interest_sets": ["Coding,Design,Marketing", "Design,Coding"],
        "degrees":       ["btech", "bca", "bdes"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["creative", "technical"],
        "goal":          ["passion"],
        "tenth_range":   (55, 90), "twelfth_range": (53, 88), "grad_range": (50, 86),
    },
    "Business Analyst": {
        "streams":       ["Commerce", "Engineering", "Arts"],
        "interest_sets": ["Business,Data,Finance", "Business,Marketing,Data",
                          "Finance,Data,Business"],
        "degrees":       ["bcom", "bba", "btech", "mba"],
        "coding":        ["none", "beginner"],
        "work_type":     ["analytical", "people"],
        "goal":          ["high_salary", "job_security"],
        "tenth_range":   (65, 95), "twelfth_range": (62, 94), "grad_range": (60, 92),
    },
    "Product Manager": {
        "streams":       ["Engineering", "Commerce", "Science"],
        "interest_sets": ["Business,Coding,Data", "Business,Marketing,Data",
                          "Business,Design,Data"],
        "degrees":       ["btech", "mba", "bcom"],
        "coding":        ["beginner", "intermediate"],
        "work_type":     ["people", "analytical"],
        "goal":          ["high_salary", "passion"],
        "tenth_range":   (72, 97), "twelfth_range": (70, 96), "grad_range": (68, 95),
    },
    "UI/UX Designer": {
        "streams":       ["Arts", "Engineering", "Science"],
        "interest_sets": ["Design,Marketing,Coding", "Design,Business", "Design,Coding"],
        "degrees":       ["bdes", "btech", "bfa", "bca"],
        "coding":        ["none", "beginner", "intermediate"],
        "work_type":     ["creative"],
        "goal":          ["passion", "high_salary"],
        "tenth_range":   (55, 92), "twelfth_range": (53, 90), "grad_range": (50, 88),
    },
    "Mobile App Developer": {
        "streams":       ["Engineering", "Science"],
        "interest_sets": ["Coding,Design,Data", "Coding,Business,Design"],
        "degrees":       ["btech", "bca", "bsc"],
        "coding":        ["intermediate", "advanced"],
        "work_type":     ["technical", "creative"],
        "goal":          ["high_salary", "passion"],
        "tenth_range":   (65, 95), "twelfth_range": (63, 94), "grad_range": (60, 92),
    },
    "Digital Marketer": {
        "streams":       ["Commerce", "Arts"],
        "interest_sets": ["Marketing,Business,Data", "Marketing,Design,Business"],
        "degrees":       ["bba", "bcom", "ba", "mba"],
        "coding":        ["none", "beginner"],
        "work_type":     ["creative", "people"],
        "goal":          ["passion", "high_salary"],
        "tenth_range":   (55, 90), "twelfth_range": (53, 88), "grad_range": (50, 85),
    },
    "Content Strategist": {
        "streams":       ["Arts", "Commerce"],
        "interest_sets": ["Marketing,Business,Design", "Marketing,Law"],
        "degrees":       ["ba", "bcom", "bjmc"],
        "coding":        ["none", "beginner"],
        "work_type":     ["creative"],
        "goal":          ["passion"],
        "tenth_range":   (55, 88), "twelfth_range": (52, 86), "grad_range": (50, 84),
    },
    "Finance Analyst": {
        "streams":       ["Commerce", "Science"],
        "interest_sets": ["Finance,Data,Business", "Finance,Business"],
        "degrees":       ["bcom", "bba", "bsc", "mba", "ca"],
        "coding":        ["none", "beginner"],
        "work_type":     ["analytical"],
        "goal":          ["high_salary", "job_security"],
        "tenth_range":   (68, 97), "twelfth_range": (65, 96), "grad_range": (62, 95),
    },
    "HR Specialist": {
        "streams":       ["Arts", "Commerce"],
        "interest_sets": ["Business,Law,Marketing", "Business,Healthcare"],
        "degrees":       ["bba", "ba", "mba"],
        "coding":        ["none"],
        "work_type":     ["people"],
        "goal":          ["job_security", "passion"],
        "tenth_range":   (55, 88), "twelfth_range": (52, 86), "grad_range": (50, 84),
    },
    "Mechanical Engineer": {
        "streams":       ["Science", "Engineering"],
        "interest_sets": ["Coding,Data,Research", "Research,Data"],
        "degrees":       ["btech", "be"],
        "coding":        ["none", "beginner"],
        "work_type":     ["technical", "analytical"],
        "goal":          ["job_security", "passion"],
        "tenth_range":   (65, 96), "twelfth_range": (62, 95), "grad_range": (60, 93),
    },
    "Civil Engineer": {
        "streams":       ["Science", "Engineering"],
        "interest_sets": ["Research,Data,Business", "Research,Data"],
        "degrees":       ["btech", "be"],
        "coding":        ["none"],
        "work_type":     ["technical", "analytical"],
        "goal":          ["job_security"],
        "tenth_range":   (62, 95), "twelfth_range": (60, 93), "grad_range": (58, 91),
    },
    "Doctor": {
        "streams":       ["Science"],
        "interest_sets": ["Healthcare,Research,Data", "Healthcare,Research"],
        "degrees":       ["mbbs", "bsc"],
        "coding":        ["none"],
        "work_type":     ["people", "analytical"],
        "goal":          ["passion", "high_salary"],
        "tenth_range":   (82, 100), "twelfth_range": (80, 100), "grad_range": (75, 100),
    },
    "Lawyer": {
        "streams":       ["Arts", "Commerce"],
        "interest_sets": ["Law,Business,Finance", "Law,Marketing,Business"],
        "degrees":       ["llb", "ba"],
        "coding":        ["none"],
        "work_type":     ["analytical", "people"],
        "goal":          ["passion", "high_salary"],
        "tenth_range":   (68, 97), "twelfth_range": (65, 96), "grad_range": (62, 95),
    },
}

# ─────────────────────────────────────────────────────────
# STEP 2 — SYNTHETIC DATA GENERATOR
# Strategy A: generate equal samples per class from the
# profile distributions above, with Gaussian noise added
# to numeric features to create variation.
# ─────────────────────────────────────────────────────────

SAMPLES_PER_CAREER = 400   # raise to 600-800 for even better accuracy


def jitter(lo, hi, noise_std=3.5):
    raw = np.random.uniform(lo, hi)
    return float(np.clip(raw + np.random.normal(0, noise_std), lo, hi))


def generate_synthetic_data(n=SAMPLES_PER_CAREER):
    rows = []
    for career, p in CAREER_PROFILES.items():
        for _ in range(n):
            rows.append({
                "stream":         random.choice(p["streams"]),
                "interests":      random.choice(p["interest_sets"]),
                "degree":         random.choice(p["degrees"]),
                "coding_skill":   random.choice(p["coding"]),
                "preferred_work": random.choice(p["work_type"]),
                "goal":           random.choice(p["goal"]),
                "tenth_pct":      round(jitter(*p["tenth_range"]), 1),
                "twelfth_pct":    round(jitter(*p["twelfth_range"]), 1),
                "grad_pct":       round(jitter(*p["grad_range"]), 1),
                "target":         career,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# STEP 3 — LOAD REAL DATA (if you have any)
# Place a career_data.csv in the same folder with columns:
# stream, interests, degree, coding_skill, preferred_work,
# goal, tenth_pct, twelfth_pct, grad_pct, target
# ─────────────────────────────────────────────────────────

def load_real_data(path="career_data.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✓ Loaded {len(df)} real samples from {path}")
        return df
    return None


# ─────────────────────────────────────────────────────────
# STEP 4 — COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────

CAT_COLS  = ["stream", "degree", "coding_skill", "preferred_work", "goal"]
NUM_COLS  = ["tenth_pct", "twelfth_pct", "grad_pct"]
TEXT_COL  = "interests"
ALL_COLS  = CAT_COLS + NUM_COLS + [TEXT_COL]


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("cat",  OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1),
                 CAT_COLS),
        ("num",  MinMaxScaler(), NUM_COLS),
        ("text", TfidfVectorizer(
                    max_features=80,
                    ngram_range=(1, 2),
                    analyzer="word",
                    token_pattern=r"[A-Za-z]+"),
                 TEXT_COL),
    ], remainder="drop")


# ─────────────────────────────────────────────────────────
# STEP 5 — SMOTE (optional, best if installed)
# Falls back gracefully if not installed.
# ─────────────────────────────────────────────────────────

def try_smote(X_prep, y):
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = sm.fit_resample(X_prep, y)
        print(f"  ✓ SMOTE applied: {len(y)} → {len(y_res)} samples")
        return X_res, y_res
    except ImportError:
        print("  ⚠ imbalanced-learn not found — using built-in strategies (A/C/D)")
        return X_prep, y


# ─────────────────────────────────────────────────────────
# STEP 6 — ENSEMBLE WRAPPER
# A single object with predict / predict_proba that Flask
# can call with a simple DataFrame. Handles missing columns
# from old User records gracefully.
# ─────────────────────────────────────────────────────────

class EnsembleCareerModel:
    """
    Wraps a preprocessor + two calibrated classifiers.
    Drop-in replacement for any sklearn pipeline.
    """
    DEFAULTS = {
        "stream":         "Science",
        "interests":      "Coding",
        "degree":         "btech",
        "coding_skill":   "beginner",
        "preferred_work": "technical",
        "goal":           "high_salary",
        "tenth_pct":      75.0,
        "twelfth_pct":    75.0,
        "grad_pct":       75.0,
    }

    def __init__(self, preprocessor, gb, rf, classes):
        self.preprocessor_ = preprocessor
        self.gb_            = gb
        self.rf_            = rf
        self.classes_       = np.array(classes)

    def _align(self, X_df):
        out = X_df.copy()
        for col, val in self.DEFAULTS.items():
            if col not in out.columns:
                out[col] = val
            else:
                out[col] = out[col].fillna(val)
        return out[ALL_COLS]

    def predict_proba(self, X_df):
        Xp = self.preprocessor_.transform(self._align(X_df))
        # 60% weight to calibrated GB (better probability estimates),
        # 40% to RF (better with imbalanced classes via class_weight)
        return 0.60 * self.gb_.predict_proba(Xp) + 0.40 * self.rf_.predict_proba(Xp)

    def predict(self, X_df):
        return self.classes_[np.argmax(self.predict_proba(X_df), axis=1)]


# ─────────────────────────────────────────────────────────
# STEP 7 — MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────

def train():
    print("\n" + "═" * 52)
    print("  CareerAI — Model Training with Imbalance Fix")
    print("═" * 52 + "\n")

    # 7a. Generate + optionally merge real data
    print("► Generating synthetic balanced data...")
    df = generate_synthetic_data(SAMPLES_PER_CAREER)

    real = load_real_data()
    if real is not None:
        # Repeat real data 3× so it has more influence than synthetic
        df = pd.concat([df, pd.concat([real] * 3)], ignore_index=True)
        print(f"  Combined total: {len(df)} rows")
    else:
        print(f"  Synthetic total: {len(df)} rows ({SAMPLES_PER_CAREER} per career)")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 7b. Class distribution
    counts = Counter(df["target"])
    print(f"\n  Class distribution  (min={min(counts.values())}  max={max(counts.values())}):")
    for k, v in sorted(counts.items(), key=lambda x: x[1]):
        bar = "▓" * (v // 25)
        print(f"    {k:<28} {v:4d}  {bar}")

    # 7c. Split
    X = df[ALL_COLS]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # 7d. Preprocess
    print("\n► Building preprocessor...")
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    # 7e. SMOTE (Strategy D)
    print("► Applying class imbalance fixes...")
    X_train_p, y_train_bal = try_smote(X_train_p, y_train)

    # 7f. Sample weights (Strategy C) for GradientBoosting
    sw = compute_sample_weight("balanced", y_train_bal)

    # 7g. Gradient Boosting + Platt calibration (Strategies C + E)
    print("► Training GradientBoosting...")
    gb_raw = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        min_samples_leaf=4,
        random_state=42,
    )
    gb_raw.fit(X_train_p, y_train_bal, sample_weight=sw)

    print("  Calibrating probabilities (Platt scaling)...")
    gb_cal = CalibratedClassifierCV(gb_raw, cv=5, method="sigmoid")
    gb_cal.fit(X_train_p, y_train_bal)

    # 7h. Random Forest with class_weight='balanced' (Strategy B)
    print("► Training RandomForest (class_weight=balanced)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    \
    rf.fit(X_train_p, y_train_bal)

    # 7i. Evaluate ensemble
    print("\n► Evaluating ensemble on held-out test set...")
    gb_proba  = gb_cal.predict_proba(X_test_p)
    rf_proba  = rf.predict_proba(X_test_p)
    ens_proba = 0.60 * gb_proba + 0.40 * rf_proba
    ens_preds = gb_cal.classes_[np.argmax(ens_proba, axis=1)]

    gb_bal_acc  = balanced_accuracy_score(y_test, gb_cal.predict(X_test_p))
    rf_bal_acc  = balanced_accuracy_score(y_test, rf.predict(X_test_p))
    ens_bal_acc = balanced_accuracy_score(y_test, ens_preds)

    print(f"\n  Balanced Accuracy:")
    print(f"    GradientBoosting + Platt : {gb_bal_acc:.4f}")
    print(f"    RandomForest (balanced)  : {rf_bal_acc:.4f}")
    print(f"    Ensemble (60/40)         : {ens_bal_acc:.4f}  ← final model")

    report = classification_report(y_test, ens_preds, zero_division=0)
    print(f"\n  Per-class report:\n{report}")

    # 7j. Cross-validation
    print("► 5-fold stratified cross-validation (RF, full data)...")
    X_full_p = preprocessor.transform(X)
    cv_scores = cross_val_score(
        rf, X_full_p, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="balanced_accuracy", n_jobs=-1,
    )
    print(f"  CV Balanced Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7k. Save report
    with open("training_report.txt", "w") as f:
        f.write("CareerAI Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples per career : {SAMPLES_PER_CAREER}\n")
        f.write(f"Total samples      : {len(df)}\n")
        f.write(f"Careers            : {len(CAREER_PROFILES)}\n\n")
        f.write("Imbalance strategies applied:\n")
        f.write("  A) Balanced synthetic data generation (equal classes)\n")
        f.write("  B) RandomForest class_weight='balanced'\n")
        f.write("  C) GradientBoosting sample_weight (balanced)\n")
        f.write("  D) SMOTE (if imbalanced-learn installed)\n")
        f.write("  E) Platt probability calibration\n")
        f.write("  F) Soft-voting ensemble (60% GB + 40% RF)\n\n")
        f.write(f"Balanced Accuracy (test set):\n")
        f.write(f"  GradientBoosting : {gb_bal_acc:.4f}\n")
        f.write(f"  RandomForest     : {rf_bal_acc:.4f}\n")
        f.write(f"  Ensemble         : {ens_bal_acc:.4f}\n\n")
        f.write(f"CV 5-fold: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write("Per-class Classification Report:\n")
        f.write(report)
    print("\n  ✓ Saved training_report.txt")

    # 7l. Save model
    model = EnsembleCareerModel(
        preprocessor = preprocessor,
        gb            = gb_cal,
        rf            = rf,
        classes       = list(gb_cal.classes_),
    )
    joblib.dump(model, "career_model.pkl")

    print("\n" + "═" * 52)
    print(f"  ✓ career_model.pkl saved")
    print(f"  ✓ Ensemble balanced accuracy: {ens_bal_acc:.4f}")
    print("  Copy career_model.pkl to your Flask root")
    print("  and restart. No other changes needed.")
    print("═" * 52 + "\n")


# ─────────────────────────────────────────────────────────
# STEP 8 — APP.PY PATCH
# The EnsembleCareerModel._align() fills missing columns
# automatically, so your existing code needs only a tiny
# update to pass the new optional fields.
# ─────────────────────────────────────────────────────────

PATCH_NOTE = """
╔══════════════════════════════════════════════════════╗
║  app.py patch (optional — improves accuracy further) ║
╚══════════════════════════════════════════════════════╝

1. Add to User model in app.py:
   degree         = db.Column(db.String(20))
   coding_skill   = db.Column(db.String(20))
   preferred_work = db.Column(db.String(20))
   goal           = db.Column(db.String(20))

2. In dashboard POST handler, add:
   user.degree         = request.form.get('degree')
   user.coding_skill   = request.form.get('coding_skill')
   user.preferred_work = request.form.get('preferred_work')
   user.goal           = request.form.get('goal')

3. In get_career_recommendations(), update X_df:
   X_df = pd.DataFrame([{
       "stream":         user.stream         or "Science",
       "interests":      user.interests      or "Coding",
       "degree":         user.degree         or "btech",
       "coding_skill":   user.coding_skill   or "beginner",
       "preferred_work": user.preferred_work or "technical",
       "goal":           user.goal           or "high_salary",
       "tenth_pct":      user.tenth_pct      or 75.0,
       "twelfth_pct":    user.twelfth_pct    if user.twelfth_pct else 75.0,
       "grad_pct":       user.grad_pct       if user.grad_pct    else 75.0,
   }])

NOTE: Missing fields are filled with defaults automatically,
so old User records without new columns still work fine.
"""


if __name__ == "__main__":
    print(PATCH_NOTE)
    train()