import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Page config & theming ---
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon=":car:",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --accent: #f97316;
        --text: #0f172a;
        --muted: #475569;
    }
    body { background: var(--bg); }
    .stApp { color: var(--text); font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: var(--text); }
    /* cards */
    .glass-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9));
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 14px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 18px 45px rgba(15,23,42,0.12);
    }
    .note-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        color: #e5e7eb;
    }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(249,115,22,0.12);
        color: var(--accent);
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.3px;
    }
    .metric-small {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached loaders ---
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# --- Preprocessing & modeling helpers ---
TARGET_COL = "sellingprice"
DROP_COLS = ["vin", "saledate"]


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        st.error(f"Kolom target '{TARGET_COL}' tidak ditemukan.")
        st.stop()
    features = df.drop(columns=[TARGET_COL], errors="ignore")
    target = df[TARGET_COL]
    return features, target


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numeric", numeric_pipeline, numeric_cols),
        ]
    )
    return preprocessor


def get_model_choices() -> Dict[str, object]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=None, random_state=42, n_jobs=-1
        ),
    }


def train_model(
    df: pd.DataFrame, model_name: str, sample_size: int, test_size: float
) -> Tuple[Pipeline, Dict[str, float]]:
    df_work = df.copy()
    for col in DROP_COLS:
        if col in df_work.columns:
            df_work = df_work.drop(columns=[col])

    if sample_size and sample_size < len(df_work):
        df_work = df_work.sample(n=sample_size, random_state=42)

    X, y = split_features_target(df_work)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    model = get_model_choices()[model_name]

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)

    metrics = {"r2": r2, "rmse": rmse, "mape": mape}
    return pipeline, metrics


def train_all_models(
    df: pd.DataFrame, sample_size: int, test_size: float
) -> Tuple[pd.DataFrame, Pipeline, Dict[str, float], str]:
    """Latih semua model yang tersedia, kembalikan tabel hasil dan pipeline terbaik."""
    results = []
    best_name = None
    best_metrics = None
    best_pipeline = None

    for name in get_model_choices().keys():
        pipeline, metrics = train_model(df, name, sample_size, test_size)
        results.append(
            {
                "Model": name,
                "R2": round(metrics["r2"], 4),
                "RMSE": round(metrics["rmse"], 3),
                "MAPE": round(metrics["mape"], 3),
            }
        )
        if best_metrics is None:
            best_name, best_metrics, best_pipeline = name, metrics, pipeline
        else:
            better_r2 = metrics["r2"] > best_metrics["r2"]
            tie_r2 = math.isclose(metrics["r2"], best_metrics["r2"], rel_tol=1e-6)
            better_rmse = metrics["rmse"] < best_metrics["rmse"]
            if better_r2 or (tie_r2 and better_rmse):
                best_name, best_metrics, best_pipeline = name, metrics, pipeline

    results_df = pd.DataFrame(results).sort_values(by=["R2", "RMSE"], ascending=[False, True])
    return results_df, best_pipeline, best_metrics, best_name


# --- Visualization helpers ---
def plot_missing(df: pd.DataFrame):
    missing = df.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("Tidak ada nilai kosong.")
        return
    fig, ax = plt.subplots(figsize=(8, min(10, 0.35 * len(missing))))
    sns.barplot(x=missing.values, y=missing.index, color="deepskyblue", edgecolor="k", ax=ax)
    ax.set_xlabel("Persen NaN (%)")
    ax.set_ylabel("Kolom")
    ax.set_title("Persentase Nilai Kosong per Kolom")
    st.pyplot(fig)


def plot_numeric_distribution(df: pd.DataFrame, columns: List[str]):
    if not columns:
        st.info("Tidak ada kolom numerik untuk ditampilkan.")
        return
    selected = st.multiselect(
        "Pilih kolom numerik", options=columns, default=columns[: min(4, len(columns))]
    )
    if not selected:
        return
    n_cols = 2
    n_rows = math.ceil(len(selected) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()
    for idx, col in enumerate(selected):
        sns.histplot(df[col].dropna(), kde=True, color="deepskyblue", ax=axes[idx])
        axes[idx].set_title(col)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    st.pyplot(fig)


def plot_correlation(df: pd.DataFrame):
    numeric_df = df.select_dtypes(exclude=["object"])
    if numeric_df.shape[1] < 2:
        st.info("Perlu minimal 2 kolom numerik untuk korelasi.")
        return
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Heatmap Korelasi Fitur Numerik")
    st.pyplot(fig)


def plot_category_counts(df: pd.DataFrame, columns: List[str]):
    if not columns:
        st.info("Tidak ada kolom kategorikal untuk ditampilkan.")
        return
    col = st.selectbox("Pilih kolom kategorikal", options=columns)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index[:20], color="darkorange", ax=ax)
    ax.set_title(f"Distribusi {col} (top 20)")
    st.pyplot(fig)


# --- Sidebar controls ---
st.sidebar.header("Pengaturan")
data_choice = st.sidebar.radio("Sumber data", ["Default car_prices.csv", "Upload CSV"])

df: pd.DataFrame | None = None
if data_choice == "Upload CSV":
    upload = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])
    if upload:
        df = load_data(upload)
else:
    try:
        df = load_data("car_prices.csv")
    except FileNotFoundError:
        st.error("File default car_prices.csv tidak ditemukan.")

sample_size = st.sidebar.number_input(
    "Batas sampel untuk training (0 = semua)", min_value=0, value=2000, step=500
)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
model_name = st.sidebar.selectbox("Model (untuk training cepat)", options=list(get_model_choices().keys()))
st.sidebar.caption("Menu 'Model Terbaik' akan melatih semua model sekaligus, mengabaikan pilihan ini.")
show_eda = st.sidebar.checkbox("Tampilkan EDA", value=True)


# --- Main layout ---
st.title("Prediksi Harga Mobil")
st.caption("EDA ringkas + training cepat + form prediksi interaktif")

if df is None:
    st.stop()

# Hero section
col_h1, col_h2 = st.columns([3, 2])
with col_h1:
    st.markdown("<span class='pill'>End-to-end</span>", unsafe_allow_html=True)
    st.markdown("## Dashboard Analitik & Prediksi Harga")
    st.markdown(
        "Visual cepat, training ringan, dan form prediksi interaktif. "
        "Gunakan batas sampel bila data besar untuk mempercepat iterasi."
    )
with col_h2:
    st.markdown(
        """
        <div class="note-card">
            <div class="metric-small">Catatan</div>
            <ul style="padding-left:16px; margin-top:4px; color: #cbd5e1;">
                <li>Jalankan <code>streamlit run streamlit_app.py</code></li>
                <li>Latih model di tab Training lalu pakai tab Prediksi</li>
                <li>Upload CSV lain di sidebar bila perlu</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Clean minimal drop for display
df_preview = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

tab_overview, tab_eda, tab_train, tab_predict, tab_best = st.tabs(
    ["Ringkasan", "EDA", "Training & Evaluasi", "Prediksi", "Model Terbaik"]
)


# --- Tab: Overview ---
with tab_overview:
    col1, col2, col3 = st.columns([1.4, 1.4, 2])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.metric("Jumlah baris", f"{len(df):,}")
        st.metric("Jumlah fitur", f"{df.shape[1]-1 if TARGET_COL in df.columns else df.shape[1]}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        if TARGET_COL in df.columns:
            st.metric("Target", TARGET_COL)
        if st.session_state.get("metrics"):
            m = st.session_state["metrics"]
            st.metric("R2 (test)", f"{m['r2']:.3f}")
            st.metric("RMSE", f"{m['rmse']:.2f}")
        else:
            st.write("Latih model untuk melihat skor.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("Snapshot data")
        st.dataframe(df_preview.head())
        st.markdown("</div>", unsafe_allow_html=True)


# --- Tab: EDA ---
with tab_eda:
    if not show_eda:
        st.info("EDA dimatikan di sidebar.")
    else:
        st.subheader("Kekosongan Data")
        plot_missing(df_preview)

        st.subheader("Distribusi Numerik")
        plot_numeric_distribution(df_preview, df_preview.select_dtypes(exclude=["object"]).columns.tolist())

        st.subheader("Korelasi")
        plot_correlation(df_preview)

        st.subheader("Distribusi Kategorikal")
        plot_category_counts(df_preview, df_preview.select_dtypes(include=["object"]).columns.tolist())


# --- Tab: Training ---
with tab_train:
    st.subheader("Training cepat")
    if st.button("Latih model", type="primary"):
        with st.spinner("Melatih model..."):
            pipeline, metrics = train_model(df, model_name, sample_size, test_size)
            st.session_state["model_pipeline"] = pipeline
            st.session_state["metrics"] = metrics
        st.success(
            f"Selesai. R2={metrics['r2']:.3f} | RMSE={metrics['rmse']:.2f} | MAPE={metrics['mape']:.2f}%"
        )
    if st.session_state.get("metrics"):
        m = st.session_state["metrics"]
        st.json(
            {
                "Model": model_name,
                "R2": round(m["r2"], 4),
                "RMSE": round(m["rmse"], 2),
                "MAPE (%)": round(m["mape"], 2),
                "Sample size": sample_size if sample_size else len(df),
                "Test size": test_size,
            }
        )


# --- Tab: Best model (multi-model evaluation) ---
with tab_best:
    st.subheader("Cari & simpan model terbaik")
    st.caption("Melatih semua model (Ridge, Lasso, RandomForest) lalu pilih R2 tertinggi (jika seri: RMSE terendah).")
    if st.button("Evaluasi semua model", type="primary"):
        with st.spinner("Melatih semua model..."):
            results_df, best_pipeline, best_metrics, best_name = train_all_models(
                df, sample_size, test_size
            )
            st.session_state["metrics_table"] = results_df
            st.session_state["best_pipeline"] = best_pipeline
            st.session_state["best_metrics"] = best_metrics
            st.session_state["best_name"] = best_name

            # simpan pickle
            filename = f"{best_name}_best_model.pkl"
            with open(filename, "wb") as f:
                pickle.dump(best_pipeline, f)
            st.session_state["best_filename"] = filename

        st.success(
            f"Model terbaik: {best_name} | R2={best_metrics['r2']:.3f} | RMSE={best_metrics['rmse']:.2f} "
            f"| disimpan ke {st.session_state['best_filename']}"
        )

    if st.session_state.get("metrics_table") is not None:
        st.markdown("**Hasil evaluasi semua model**")
        st.dataframe(st.session_state["metrics_table"], use_container_width=True)

    if st.session_state.get("best_filename"):
        st.info(f"File tersimpan: {st.session_state['best_filename']}")
        st.caption("Gunakan di tab Prediksi dengan memilih opsi model tersimpan (jika Anda menambahkan loader), atau langsung pakai session pipeline.")


# --- Tab: Predict ---
with tab_predict:
    st.subheader("Prediksi harga")
    # gunakan model_pipeline dari training cepat atau best_pipeline jika ada
    active_model = st.session_state.get("model_pipeline") or st.session_state.get("best_pipeline")
    if active_model is None:
        st.info("Latih model dulu di tab 'Training & Evaluasi' atau 'Model Terbaik'.")
        st.stop()

    features_df, _ = split_features_target(df.drop(columns=[c for c in DROP_COLS if c in df.columns]))
    cat_cols = features_df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = features_df.select_dtypes(exclude=["object"]).columns.tolist()

    with st.form("prediction_form"):
        cat_inputs = {}
        num_inputs = {}
        if cat_cols:
            st.markdown("**Fitur kategorikal**")
            for col in cat_cols:
                options = sorted(features_df[col].dropna().unique().tolist())
                cat_inputs[col] = st.selectbox(col, options=options or ["others"], index=0)
        if num_cols:
            st.markdown("**Fitur numerik**")
            cols = st.columns(2)
            for idx, col in enumerate(num_cols):
                median = float(features_df[col].median()) if not features_df[col].empty else 0.0
                num_inputs[col] = cols[idx % 2].number_input(col, value=median)
        submitted = st.form_submit_button("Hitung Harga", type="primary")

    if submitted:
        input_df = pd.DataFrame([{**cat_inputs, **num_inputs}])
        pred = float(active_model.predict(input_df)[0])
        st.success(f"Perkiraan harga: ${pred:,.2f}")
