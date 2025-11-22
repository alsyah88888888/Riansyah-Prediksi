import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

st.title("🚗 Dashboard Prediksi Harga Mobil")
st.markdown("Analisis Data dan Pemodelan Machine Learning")

# --- 1. LOAD DATA & CLEANING (CACHED) ---
@st.cache_data
def load_and_clean_data():
    # Load Data
    dataset = pd.read_csv('car_prices.csv') # Pastikan nama file sesuai
    
    # Drop kolom tidak terpakai
    if 'vin' in dataset.columns:
        dataset = dataset.drop(columns=['vin', 'saledate'], errors='ignore')
    
    # Cleaning Float ke NaN
    cols_to_check = ['make', 'model', 'trim', 'body', 'transmission', 'color', 'interior']
    for col in cols_to_check:
        dataset[col] = dataset[col].apply(lambda x: np.nan if isinstance(x, float) else x)
    
    # Imputasi Numerik (Median)
    dataset_num = dataset.select_dtypes(exclude=['object'])
    cols_to_fill_num = ['condition', 'odometer', 'mmr', 'sellingprice']
    for col in cols_to_fill_num:
        if col in dataset_num.columns:
            median_val = dataset_num[col].median()
            dataset_num[col] = dataset_num[col].fillna(median_val)
            dataset[col] = dataset[col].fillna(median_val) # Update dataset utama juga
            
    # Imputasi Kategorikal (Others)
    dataset_categ = dataset.select_dtypes(include=['object'])
    cols_to_fill_cat = ['make', 'model', 'trim', 'body', 'transmission', 'color', 'interior']
    for col in cols_to_fill_cat:
        if col in dataset_categ.columns:
            dataset[col] = dataset[col].fillna('others')
            
    # Cleaning khusus Transmission
    dataset['transmission'] = dataset['transmission'].replace(['sedan', 'Sedan'], np.nan)
    dataset['transmission'] = dataset['transmission'].fillna('others')
    
    return dataset

# --- 2. FEATURE ENGINEERING (CACHED) ---
@st.cache_data
def process_features(dataset):
    # Copy dataset agar tidak merusak yang asli
    df_processed = dataset.copy()
    
    # Simpan sellingprice
    sellingprice = df_processed['sellingprice']
    
    # One-Hot Encoding untuk Transmission
    if 'transmission' in df_processed.columns:
        ohe = OneHotEncoder(sparse_output=False, drop=None)
        transmission_encoded = ohe.fit_transform(df_processed[['transmission']])
        transmission_df = pd.DataFrame(
            transmission_encoded,
            columns=[f"transmission_{cat}" for cat in ohe.categories_[0]],
            index=df_processed.index
        )
        df_processed = pd.concat([df_processed.drop(columns=['transmission']), transmission_df], axis=1)
        
    # Label Encoding untuk kolom kategorikal lainnya
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
    # Scaling (kecuali target)
    scaler = StandardScaler()
    cols_to_scale = df_processed.columns.drop('sellingprice')
    df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
    
    # Kembalikan sellingprice (target)
    df_processed['sellingprice'] = sellingprice
    
    return df_processed

# --- EKSEKUSI UTAMA ---

try:
    # Load Data
    with st.spinner('Sedang memuat data...'):
        df = load_and_clean_data()
        df_ready = process_features(df)
        
    # Membuat Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 EDA", "🤖 Modelling"])

    # --- TAB 1: DATA OVERVIEW ---
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Shape Data:", df.shape)
            st.write("Tipe Data:")
            st.write(df.dtypes)
        with col2:
            st.write("Statistik Deskriptif:")
            st.write(df.describe())

    # --- TAB 2: EDA ---
    with tab2:
        st.subheader("Distribusi Data Numerik")
        
        num_cols = df.select_dtypes(include=['number']).columns
        # Hapus kolom hasil one-hot encoding dari visualisasi agar tidak penuh
        viz_cols = [c for c in num_cols if not c.startswith('transmission_')]
        
        selected_col = st.selectbox("Pilih kolom untuk histogram:", viz_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, ax=ax, color="deepskyblue")
        st.pyplot(fig)
        
        st.subheader("Korelasi Heatmap")
        if st.checkbox("Tampilkan Heatmap (Mungkin lambat)"):
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[viz_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)

    # --- TAB 3: MODELLING ---
    with tab3:
        st.warning("⚠️ Training model menggunakan GridSearchCV memakan waktu lama. Klik tombol di bawah untuk memulai.")
        
        if st.button("Mulai Training Model"):
            # Splitting Data
            X = df_ready.drop(columns=['sellingprice'])
            y = df_ready['sellingprice']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            models_to_train = {
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
                "LightGBM": LGBMRegressor(random_state=42),
                "Linear Regression": LinearRegression()
            }
            
            # Parameter Grid (Disederhanakan agar tidak Timeout di Streamlit Cloud)
            # Kita kurangi jumlah iterasi/cv agar cepat
            params = {
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.1, 1.0, 10.0]},
                "XGBoost": {"n_estimators": [50, 100], "max_depth": [3], "learning_rate": [0.1]},
                "LightGBM": {"n_estimators": [50, 100], "num_leaves": [31]},
                "Linear Regression": {"fit_intercept": [True, False]}
            }
            
            best_model_overall = None
            best_r2_overall = -1
            best_name_overall = ""
            
            total_models = len(models_to_train)
            current_step = 0

            # Loop Training
            for name, model in models_to_train.items():
                status_text.text(f"Training {name}...")
                
                # Handle NaN jika masih ada sisa (Safety net)
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)

                grid = GridSearchCV(model, params[name], cv=3, n_jobs=-1, scoring='r2')
                grid.fit(X_train, y_train)
                
                y_pred = grid.best_estimator_.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                results.append({"Model": name, "R2": r2, "RMSE": rmse})
                
                if r2 > best_r2_overall:
                    best_r2_overall = r2
                    best_model_overall = grid.best_estimator_
                    best_name_overall = name
                
                current_step += 1
                progress_bar.progress(current_step / total_models)

            status_text.text("Training Selesai!")
            
            # Tampilkan Hasil
            st.subheader("Hasil Evaluasi Model")
            results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
            st.dataframe(results_df)
            
            st.success(f"🏆 Model Terbaik: **{best_name_overall}** dengan R²: {best_r2_overall:.4f}")
            
            # Plot Actual vs Predicted untuk model terbaik
            st.subheader(f"Prediksi vs Aktual ({best_name_overall})")
            y_pred_best = best_model_overall.predict(X_test)
            
            fig_res, ax_res = plt.subplots(figsize=(8, 6))
            ax_res.scatter(y_test, y_pred_best, color="deepskyblue", alpha=0.5)
            ax_res.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "darkorange", lw=2)
            ax_res.set_xlabel("Actual Price")
            ax_res.set_ylabel("Predicted Price")
            ax_res.set_title(f"Actual vs Predicted ({best_name_overall})")
            st.pyplot(fig_res)
            
            # Simpan Model
            # (Optional di Streamlit Cloud karena file akan hilang saat reboot, 
            # tapi bagus untuk fitur download)
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model_overall, f)
            
            with open("best_model.pkl", "rb") as f:
                st.download_button("Download Best Model (.pkl)", f, file_name="best_model_car_price.pkl")

except FileNotFoundError:
    st.error("❌ File 'car_prices.csv' tidak ditemukan. Pastikan file sudah diupload ke GitHub dan namanya sesuai (case-sensitive).")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")