import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor


# =======================
# 1. Load dataset
# =======================
@st.cache_data
def load_data(path="car_prices.csv"):
    df = pd.read_csv(path)

    feature_cols = [
        "year",
        "condition",
        "odometer",
        "mmr",
        "make",
        "model",
        "trim",
        "body",
        "state",
        "color",
        "interior",
        "transmission"
    ]

    target_col = "sellingprice"

    numeric_cols = ["year", "condition", "odometer", "mmr"]
    cat_cols = ["make", "model", "trim", "body", "state", "color", "interior", "transmission"]

    df = df[feature_cols + [target_col]].copy()

    # Handle missing numerik
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Handle missing kategori
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    df = df.dropna(subset=[target_col])

    return df, feature_cols, target_col, numeric_cols, cat_cols


# =======================
# 2. Train LightGBM model
# =======================
@st.cache_resource
def train_model(df, feature_cols, target_col, numeric_cols, cat_cols):
    data = df.copy()

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, encoders, r2, mae


# =======================
# 3. STREAMLIT UI
# =======================
st.title("Prediksi Harga Mobil - LightGBM")

df, feature_cols, target_col, numeric_cols, cat_cols = load_data()

model, encoders, r2, mae = train_model(df, feature_cols, target_col, numeric_cols, cat_cols)

st.success("Model dilatih!")
st.write(f"R² (test): **{r2:.4f}**")
st.write(f"MAE (test): **{mae:,.2f}**")

st.markdown("---")

st.header("Input Fitur Mobil")


def col_min(c):
    return float(df[c].min())


def col_max(c):
    return float(df[c].max())


# =======================
# 4. INPUT FORM
# =======================
with st.form("form"):
    year = st.slider("Year", int(col_min("year")), int(col_max("year")), int(df["year"].median()))

    condition = st.slider("Condition", float(col_min("condition")), float(col_max("condition")),
                          float(df["condition"].median()), step=1.0)

    odometer = st.number_input("Odometer", float(col_min("odometer")), float(col_max("odometer")),
                               float(df["odometer"].median()))

    mmr = st.number_input("MMR", float(col_min("mmr")), float(col_max("mmr")), float(df["mmr"].median()))

    make = st.selectbox("Make", list(encoders["make"].classes_))
    model_name = st.selectbox("Model", list(encoders["model"].classes_))
    trim = st.selectbox("Trim", list(encoders["trim"].classes_))
    body = st.selectbox("Body", list(encoders["body"].classes_))
    state = st.selectbox("State", list(encoders["state"].classes_))
    color = st.selectbox("Color", list(encoders["color"].classes_))
    interior = st.selectbox("Interior", list(encoders["interior"].classes_))
    transmission = st.selectbox("Transmission", list(encoders["transmission"].classes_))

    submit = st.form_submit_button("Prediksi Harga")


# =======================
# 5. PREDIKSI
# =======================
if submit:

    def encode(col, val):
        return int(encoders[col].transform([val])[0])

    row = [
        year,
        condition,
        odometer,
        mmr,
        encode("make", make),
        encode("model", model_name),
        encode("trim", trim),
        encode("body", body),
        encode("state", state),
        encode("color", color),
        encode("interior", interior),
        encode("transmission", transmission)
    ]

    pred = model.predict(np.array(row).reshape(1, -1))[0]

    st.subheader("Hasil Prediksi")
    st.success(f"Perkiraan harga mobil: **${pred:,.2f}**")
