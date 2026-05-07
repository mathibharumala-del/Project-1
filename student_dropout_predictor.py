"""
╔══════════════════════════════════════════════════════════════════╗
║        STUDENT DROPOUT RISK PREDICTOR                           ║
║        Dataset : Student Dropout Prediction Dataset             ║
║                  (Kaggle – meharshanali, Feb 2026)              ║
║        Stack   : pandas · scikit-learn · xgboost · shap ·       ║
║                  matplotlib · seaborn · streamlit               ║
╚══════════════════════════════════════════════════════════════════╝

SETUP (run once):
    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn streamlit

HOW TO RUN:
    ① Training + evaluation:
        python student_dropout_predictor.py

    ② Interactive dashboard:
        streamlit run student_dropout_predictor.py

    ③ Put dataset.csv in the same folder, or update CSV_PATH below.
"""

# ─────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────
import os, sys, warnings, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, ConfusionMatrixDisplay
)
from sklearn.impute          import SimpleImputer

# Optional — install with:  pip install xgboost shap
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[INFO] xgboost not found. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] shap not found. Install with: pip install shap")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
CSV_PATH     = "dataset.csv"   # ← path to your downloaded CSV
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MODEL_PATH   = "dropout_model.pkl"
SCALER_PATH  = "dropout_scaler.pkl"

# ─────────────────────────────────────────────────────────────
#  1. DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ '{path}' not found.\n"
            "   Download from: https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset\n"
            "   Then place dataset.csv in the same folder as this script.\n"
        )
    df = pd.read_csv(path)
    print(f"\n✅  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Target distribution:\n{df['Target'].value_counts().to_string()}\n")
    return df

# ─────────────────────────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 6 derived features that capture student trajectory signals
    not visible in the raw columns alone.
    """
    df = df.copy()

    # Academic momentum
    df['grade_trend'] = (
        df['Curricular units 2nd sem (grade)'] -
        df['Curricular units 1st sem (grade)']
    )
    df['avg_grade'] = (
        df['Curricular units 1st sem (grade)'] +
        df['Curricular units 2nd sem (grade)']
    ) / 2

    # Approval rates (how many enrolled units were actually passed)
    df['approval_rate_1st'] = (
        df['Curricular units 1st sem (approved)'] /
        (df['Curricular units 1st sem (enrolled)'] + 1e-5)
    )
    df['approval_rate_2nd'] = (
        df['Curricular units 2nd sem (approved)'] /
        (df['Curricular units 2nd sem (enrolled)'] + 1e-5)
    )

    # Total academic output
    df['total_approved'] = (
        df['Curricular units 1st sem (approved)'] +
        df['Curricular units 2nd sem (approved)']
    )

    # Composite financial stability flag
    df['financially_stable'] = (
        (df['Tuition fees up to date'] == 1) &
        (df['Debtor'] == 0)
    ).astype(int)

    return df

# ─────────────────────────────────────────────────────────────
#  3. PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Encode target, split features, stratified train/test split."""
    le = LabelEncoder()
    df = df.copy()
    df['target_enc'] = le.fit_transform(df['Target'])
    df['is_dropout']  = (df['Target'] == 'Dropout').astype(int)

    feature_cols = [c for c in df.columns if c not in ['Target', 'target_enc', 'is_dropout']]
    X = df[feature_cols]
    y = df['target_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, le, feature_cols, X

# ─────────────────────────────────────────────────────────────
#  4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────
def build_models():
    """Return dict of candidate classifiers."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=12,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, random_state=RANDOM_STATE
        ),
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=-1
        )
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    models   = build_models()
    results  = {}

    print("─" * 55)
    print(f"  {'Model':<25}  {'F1 (wt)':>8}  {'AUC (wt)':>9}")
    print("─" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        f1  = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(
            pd.get_dummies(y_test), y_prob,
            multi_class='ovr', average='weighted'
        )
        results[name] = {
            'model': model, 'f1': f1, 'auc': auc,
            'y_pred': y_pred, 'y_prob': y_prob
        }
        print(f"  {name:<25}  {f1:>8.4f}  {auc:>9.4f}")

    print("─" * 55)
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\n  🏆 Best model: {best_name}")
    print(f"\n  Classification Report — {best_name}")
    print(classification_report(
        y_test, results[best_name]['y_pred'],
        target_names=le.classes_
    ))
    return results, best_name

# ─────────────────────────────────────────────────────────────
#  5. EDA PLOTS
# ─────────────────────────────────────────────────────────────
def plot_eda(df: pd.DataFrame, out_dir: str = "eda_plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── Plot 1: Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Target'].value_counts()
    colors = ['#E24B4A', '#378ADD', '#639922']
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax.set_title('Target class distribution', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Number of students')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/01_class_distribution.png', dpi=150)
    plt.close()

    # ── Plot 2: Age at enrollment by outcome
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in zip(['Graduate', 'Dropout', 'Enrolled'],
                             ['#639922', '#E24B4A', '#378ADD']):
        subset = df[df['Target'] == label]['Age at enrollment']
        ax.hist(subset, bins=30, alpha=0.55, label=label, color=color, edgecolor='white')
    ax.set_title('Age at enrollment by outcome', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Age at enrollment')
    ax.set_ylabel('Count')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/02_age_distribution.png', dpi=150)
    plt.close()

    # ── Plot 3: Grade distribution by outcome (boxplot)
    fig, ax = plt.subplots(figsize=(7, 4))
    grade_data = [df[df['Target'] == t]['avg_grade'].values
                  for t in ['Dropout', 'Enrolled', 'Graduate']]
    bp = ax.boxplot(grade_data, labels=['Dropout', 'Enrolled', 'Graduate'],
                    patch_artist=True, medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], ['#E24B4A', '#378ADD', '#639922']):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title('Average grade by outcome', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Average grade (both semesters)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/03_grade_by_outcome.png', dpi=150)
    plt.close()

    # ── Plot 4: Feature correlation heatmap
    key_feats = [
        'avg_grade', 'grade_trend', 'approval_rate_1st', 'approval_rate_2nd',
        'total_approved', 'Age at enrollment', 'Tuition fees up to date',
        'Debtor', 'Scholarship holder', 'Unemployment rate', 'GDP', 'is_dropout'
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df[key_feats].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.4, ax=ax, annot_kws={'size': 8})
    ax.set_title('Feature correlation matrix (key features)', fontsize=13,
                 fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/04_correlation_heatmap.png', dpi=150)
    plt.close()

    # ── Plot 5: Financial factors vs outcome
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(
        axes,
        ['Tuition fees up to date', 'Debtor', 'Scholarship holder'],
        ['Tuition fees up to date', 'Debtor status', 'Scholarship holder']
    ):
        cross = pd.crosstab(df[col], df['Target'], normalize='index') * 100
        cross[['Dropout', 'Enrolled', 'Graduate']].plot(
            kind='bar', ax=ax, color=['#E24B4A', '#378ADD', '#639922'],
            edgecolor='white', width=0.7
        )
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('% of students')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    plt.suptitle('Financial factors vs dropout outcome', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/05_financial_factors.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ EDA plots saved to '{out_dir}/'")

# ─────────────────────────────────────────────────────────────
#  6. MODEL PLOTS
# ─────────────────────────────────────────────────────────────
def plot_model_results(results, best_name, y_test, le,
                       feature_cols, out_dir="model_plots"):
    os.makedirs(out_dir, exist_ok=True)
    best = results[best_name]

    # ── Plot 6: Model comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    f1s   = [results[n]['f1']  for n in names]
    aucs  = [results[n]['auc'] for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, f1s,  w, label='Weighted F1',  color='#378ADD', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, aucs, w, label='Weighted AUC', color='#639922', alpha=0.85, edgecolor='white')
    for i, (f, a) in enumerate(zip(f1s, aucs)):
        ax.text(i - w/2, f + 0.004, f'{f:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + w/2, a + 0.004, f'{a:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel('Score')
    ax.set_title('Model comparison — Weighted F1 & AUC', fontsize=13,
                 fontweight='bold', pad=12)
    ax.legend(); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/06_model_comparison.png', dpi=150)
    plt.close()

    # ── Plot 7: Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm   = confusion_matrix(y_test, best['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion matrix — {best_name}', fontsize=13,
                 fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/07_confusion_matrix.png', dpi=150)
    plt.close()

    # ── Plot 8: Feature importance
    model = best['model']
    if hasattr(model, 'feature_importances_'):
        fi     = pd.Series(model.feature_importances_, index=feature_cols)
        fi_top = fi.nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        colors_fi = ['#E24B4A' if v > fi_top.quantile(0.75) else '#378ADD'
                     for v in fi_top.values]
        fi_top.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
        ax.set_title(f'Top 15 feature importances — {best_name}',
                     fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Importance score')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/08_feature_importance.png', dpi=150)
        plt.close()

    # ── Plot 9: Dropout probability distribution
    le_classes = list(le.classes_)
    dropout_idx = le_classes.index('Dropout')
    dropout_probs = best['y_prob'][:, dropout_idx]
    actual_dropout = (y_test == dropout_idx).astype(int).values

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dropout_probs[actual_dropout == 1], bins=30, alpha=0.65,
            color='#E24B4A', label='Actual Dropout', density=True)
    ax.hist(dropout_probs[actual_dropout == 0], bins=30, alpha=0.65,
            color='#639922', label='Not Dropout', density=True)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.2, label='Threshold 0.5')
    ax.set_title('Dropout risk score distribution', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_xlabel('Predicted dropout probability')
    ax.set_ylabel('Density')
    ax.legend(); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/09_risk_score_distribution.png', dpi=150)
    plt.close()

    print(f"  ✅ Model plots saved to '{out_dir}/'")

# ─────────────────────────────────────────────────────────────
#  7. SHAP EXPLAINABILITY  (requires: pip install shap)
# ─────────────────────────────────────────────────────────────
def plot_shap(model, X_test_s, feature_cols, le, out_dir="model_plots"):
    if not SHAP_AVAILABLE:
        print("  ⚠️  SHAP not available. Run: pip install shap")
        return

    os.makedirs(out_dir, exist_ok=True)
    X_sample = X_test_s[:200]   # sample for speed
    X_sample_df = pd.DataFrame(X_sample, columns=feature_cols)

    print("  Computing SHAP values (this may take ~30s)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_df)

    # shap_values can be:
    #   • list of arrays  [classes × (n_samples, n_features)]  — older SHAP
    #   • 3D array        (n_samples, n_features, n_classes)   — newer SHAP
    # Normalise to always get a 2D array for class 0 (Dropout)
    if isinstance(shap_values, list):
        # Older SHAP: list of (n_samples, n_features) arrays
        sv_dropout = shap_values[0]                      # class 0 = Dropout
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Newer SHAP: (n_samples, n_features, n_classes)
        sv_dropout = shap_values[:, :, 0]               # class 0 = Dropout
    else:
        # Already 2D (binary or single-output) — use as-is
        sv_dropout = shap_values

    # Summary plot — Dropout class
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        sv_dropout, X_sample_df,
        feature_names=feature_cols, show=False
    )
    plt.title('SHAP feature impact — Dropout class', fontsize=13,
              fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/10_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ SHAP plot saved to '{out_dir}/10_shap_summary.png'")

# ─────────────────────────────────────────────────────────────
#  8. SAVE MODEL
# ─────────────────────────────────────────────────────────────
def save_model(model, scaler):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✅ Model saved → {MODEL_PATH}")
    print(f"  ✅ Scaler saved → {SCALER_PATH}")

# ─────────────────────────────────────────────────────────────
#  9. STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────
def run_dashboard():
    """
    Launch with:  streamlit run student_dropout_predictor.py
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="Student Dropout Risk Predictor",
        page_icon="🎓",
        layout="wide"
    )

    @st.cache_resource
    def load_model_cached():
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    st.title("🎓 Student Dropout Risk Predictor")
    st.markdown(
        "Enter a student's details below to get their dropout risk score "
        "and understand which factors are driving the prediction."
    )

    # ── Sidebar: student input
    st.sidebar.header("Student Profile")

    age           = st.sidebar.slider("Age at enrollment", 17, 70, 20)
    gender        = st.sidebar.selectbox("Gender", ["Male (1)", "Female (0)"])
    scholarship   = st.sidebar.selectbox("Scholarship holder?", ["No (0)", "Yes (1)"])
    debtor        = st.sidebar.selectbox("Has outstanding debt?", ["No (0)", "Yes (1)"])
    tuition_ok    = st.sidebar.selectbox("Tuition fees up to date?", ["Yes (1)", "No (0)"])
    daytime       = st.sidebar.selectbox("Attendance", ["Daytime (1)", "Evening (0)"])

    st.sidebar.subheader("1st Semester")
    enrolled_1    = st.sidebar.number_input("Units enrolled",  0, 20, 6,  key="e1")
    approved_1    = st.sidebar.number_input("Units approved",  0, 20, 5,  key="a1")
    grade_1       = st.sidebar.slider("Average grade", 0.0, 20.0, 12.0, key="g1")

    st.sidebar.subheader("2nd Semester")
    enrolled_2    = st.sidebar.number_input("Units enrolled",  0, 20, 6,  key="e2")
    approved_2    = st.sidebar.number_input("Units approved",  0, 20, 5,  key="a2")
    grade_2       = st.sidebar.slider("Average grade", 0.0, 20.0, 12.0, key="g2")

    st.sidebar.subheader("Economic Context")
    unemp         = st.sidebar.slider("Unemployment rate", 7.0, 17.0, 11.0)
    gdp           = st.sidebar.slider("GDP growth rate", -5.0, 4.0, 0.5)

    # Build feature row (must match training column order)
    gender_val      = int(gender.split("(")[1].replace(")", ""))
    scholarship_val = int(scholarship.split("(")[1].replace(")", ""))
    debtor_val      = int(debtor.split("(")[1].replace(")", ""))
    tuition_val     = int(tuition_ok.split("(")[1].replace(")", ""))
    daytime_val     = int(daytime.split("(")[1].replace(")", ""))

    avg_grade       = (grade_1 + grade_2) / 2
    grade_trend     = grade_2 - grade_1
    approval_r1     = approved_1 / (enrolled_1 + 1e-5)
    approval_r2     = approved_2 / (enrolled_2 + 1e-5)
    total_approved  = approved_1 + approved_2
    fin_stable      = int(tuition_val == 1 and debtor_val == 0)

    # Placeholder zeros for columns not collected in the UI
    # (marital status, application mode, course, etc.)
    input_row = {
        'Marital status': 1,
        'Application mode': 1,
        'Application order': 1,
        'Course': 1,
        'Daytime/evening attendance': daytime_val,
        'Previous qualification': 1,
        'Nacionality': 1,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 1,
        "Father's occupation": 1,
        'Displaced': 0,
        'Educational special needs': 0,
        'Debtor': debtor_val,
        'Tuition fees up to date': tuition_val,
        'Gender': gender_val,
        'Scholarship holder': scholarship_val,
        'Age at enrollment': age,
        'International': 0,
        'Curricular units 1st sem (credited)': 0,
        'Curricular units 1st sem (enrolled)': enrolled_1,
        'Curricular units 1st sem (evaluations)': enrolled_1,
        'Curricular units 1st sem (approved)': approved_1,
        'Curricular units 1st sem (grade)': grade_1,
        'Curricular units 1st sem (without evaluations)': 0,
        'Curricular units 2nd sem (credited)': 0,
        'Curricular units 2nd sem (enrolled)': enrolled_2,
        'Curricular units 2nd sem (evaluations)': enrolled_2,
        'Curricular units 2nd sem (approved)': approved_2,
        'Curricular units 2nd sem (grade)': grade_2,
        'Curricular units 2nd sem (without evaluations)': 0,
        'Unemployment rate': unemp,
        'Inflation rate': 1.4,
        'GDP': gdp,
        # Engineered features
        'grade_trend': grade_trend,
        'avg_grade': avg_grade,
        'approval_rate_1st': approval_r1,
        'approval_rate_2nd': approval_r2,
        'total_approved': total_approved,
        'financially_stable': fin_stable,
    }

    try:
        model, scaler = load_model_cached()
        X_in  = pd.DataFrame([input_row])
        X_s   = scaler.transform(X_in)
        proba = model.predict_proba(X_s)[0]
        classes = ['Dropout', 'Enrolled', 'Graduate']   # LabelEncoder order
        dropout_prob = proba[0]

        # ── Risk gauge
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_pct = f"{dropout_prob * 100:.1f}%"
            if dropout_prob > 0.65:
                st.error(f"🔴 **HIGH RISK** — {risk_pct} dropout probability")
            elif dropout_prob > 0.35:
                st.warning(f"🟡 **MEDIUM RISK** — {risk_pct} dropout probability")
            else:
                st.success(f"🟢 **LOW RISK** — {risk_pct} dropout probability")

        # ── Probability bars
        st.subheader("Outcome probabilities")
        for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

        # ── Key insight
        st.subheader("Key risk factors for this student")
        insights = []
        if avg_grade < 10:
            insights.append("📉 Average grade below 10 — strong dropout predictor")
        if grade_trend < -2:
            insights.append("📉 Grades declining between semesters")
        if approval_r1 < 0.5 or approval_r2 < 0.5:
            insights.append("⚠️ Low unit approval rate — academic difficulty")
        if debtor_val == 1:
            insights.append("💳 Outstanding debt — financial risk factor")
        if tuition_val == 0:
            insights.append("💸 Tuition not up to date — strong dropout signal")
        if not insights:
            insights.append("✅ No major risk factors detected for this student")
        for i in insights:
            st.markdown(f"- {i}")

    except FileNotFoundError:
        st.warning(
            "⚠️ No trained model found yet.\n\n"
            "Run `python student_dropout_predictor.py` first to train and save the model."
        )

    # ── Show saved plots if they exist
    st.subheader("Model Insights")
    plot_cols = st.columns(2)
    plot_files = [
        ("eda_plots/01_class_distribution.png",    "Class distribution"),
        ("model_plots/06_model_comparison.png",    "Model comparison"),
        ("model_plots/07_confusion_matrix.png",    "Confusion matrix"),
        ("model_plots/08_feature_importance.png",  "Feature importance"),
        ("eda_plots/03_grade_by_outcome.png",      "Grades by outcome"),
        ("model_plots/09_risk_score_distribution.png", "Risk score distribution"),
    ]
    for i, (path, caption) in enumerate(plot_files):
        if os.path.exists(path):
            with plot_cols[i % 2]:
                st.image(path, caption=caption, use_column_width=True)


# ─────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  STUDENT DROPOUT RISK PREDICTOR")
    print("=" * 55)

    # 1. Load
    df = load_data(CSV_PATH)

    # 2. Feature engineering
    df = engineer_features(df)
    df['is_dropout'] = (df['Target'] == 'Dropout').astype(int)

    # 3. EDA plots
    print("\n[1/4] Running EDA...")
    plot_eda(df)

    # 4. Preprocess + split
    print("\n[2/4] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, le, feature_cols, X = preprocess(df)

    # 5. Train & evaluate
    print("\n[3/4] Training models...\n")
    results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test, le)

    # 6. Model plots
    print("\n[4/4] Generating model plots...")
    plot_model_results(results, best_name, y_test, le, feature_cols)

    # 7. SHAP (if available)
    if SHAP_AVAILABLE and hasattr(results[best_name]['model'], 'feature_importances_'):
        plot_shap(results[best_name]['model'], X_test, feature_cols, le)

    # 8. Save best model
    save_model(results[best_name]['model'], scaler)

    print("\n" + "=" * 55)
    print("  ✅  TRAINING COMPLETE")
    print("  ─────────────────────────────────────────────")
    print(f"  Best model : {best_name}")
    print(f"  F1 score   : {results[best_name]['f1']:.4f}")
    print(f"  AUC score  : {results[best_name]['auc']:.4f}")
    print("  ─────────────────────────────────────────────")
    print("  Launch dashboard:  streamlit run student_dropout_predictor.py")
    print("=" * 55 + "\n")


# Detect whether we're running under Streamlit or as a script
if __name__ == "__main__":
    # If launched via `streamlit run`, sys.argv[0] contains "streamlit"
    # or the script is invoked differently
    import sys
    if "streamlit" in sys.modules or any("streamlit" in a for a in sys.argv):
        run_dashboard()
    else:
        main()
