import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import pyswarms as ps
import shap

# Konfigurasi Halaman
st.set_page_config(layout="wide", page_title="Medical ML Dashboard")
st.title("📊 EDA + Feature Selection + SMOTE + PSO + Prediksi Pasien")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    target_col = st.selectbox("Pilih Kolom Target (Label)", df.columns)

    if target_col:
        # ================= Preprocessing Data =================
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode Target
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        num_classes = len(np.unique(y))
        
        # Identifikasi tipe kolom
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        # Imputasi & Encoding Fitur
        for col in num_cols:
            X[col] = X[col].fillna(X[col].median())

        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
            le_feat = LabelEncoder()
            X[col] = le_feat.fit_transform(X[col].astype(str))

        # Standarisasi (Simpan scaler untuk inferensi nanti)
        scaler = StandardScaler()
        if len(num_cols) > 0:
            X[num_cols] = scaler.fit_transform(X[num_cols])

        # Split Data
        if num_classes < 2:
            st.error("Target hanya memiliki 1 kelas. Minimal butuh 2 kelas.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ================= Feature Selection =================
        st.header("1. Feature Selection (ANOVA)")
        max_k = min(15, X.shape[1])
        k = st.slider("Jumlah fitur terbaik untuk digunakan", 1, max_k, min(10, max_k))

        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        
        selected_features = X_train.columns[selector.get_support()]
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        st.write(f"Fitur terpilih: {', '.join(selected_features)}")

        # ================= SMOTE =================
        st.header("2. SMOTE (Oversampling)")
        class_counts = Counter(y_train)
        minority = min(class_counts.values())

        if minority > 1:
            k_neighbors = min(5, minority - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_sm, y_train_sm = smote.fit_resample(X_train_sel, y_train)
            st.write("Distribusi kelas setelah SMOTE:", Counter(y_train_sm))
        else:
            st.warning("SMOTE dilewati (data minoritas terlalu sedikit).")
            X_train_sm, y_train_sm = X_train_sel, y_train

        # ================= Modeling =================
        st.header("3. Model Training & Evaluation")
        models = {
            "SVM": SVC(kernel="rbf", probability=True),
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier()
        }

        results = []
        for name, model in models.items():
            try:
                model.fit(X_train_sm, y_train_sm)
                y_pred = model.predict(X_test_sel)
                acc = accuracy_score(y_test, y_pred)
                results.append({"Model": name, "Test Accuracy": acc})
            except Exception as e:
                continue

        results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
        st.table(results_df)

        # ================= PSO Optimization =================
        st.header("4. SVM Optimization with PSO")
        with st.spinner("Menjalankan optimasi Particle Swarm..."):
            def objective(params):
                costs = []
                for p in params:
                    # p[0] = C, p[1] = Gamma
                    m = SVC(C=p[0], gamma=p[1], kernel="rbf")
                    score = cross_val_score(m, X_train_sm, y_train_sm, cv=3).mean()
                    costs.append(1 - score)
                return np.array(costs)

            bounds = (np.array([0.01, 0.0001]), np.array([50.0, 1.0]))
            optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, 
                                                options={'c1':0.5, 'c2':0.3, 'w':0.9}, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(objective, iters=5)
            
            st.success(f"Parameter Terbaik Ditemukan -> C: {best_pos[0]:.4f}, Gamma: {best_pos[1]:.4f}")

        # Final Optimized Model
        svm_opt = SVC(C=best_pos[0], gamma=best_pos[1], kernel="rbf", probability=True).fit(X_train_sm, y_train_sm)

        # ================= SHAP =================
        st.header("5. SHAP Interpretability")
        try:
            rf_shap = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_sm, y_train_sm)
            explainer = shap.TreeExplainer(rf_shap)
            shap_test_data = X_test_sel.head(100) if len(X_test_sel) > 100 else X_test_sel
            shap_values = explainer.shap_values(shap_test_data)
            
            fig_shap, ax_shap = plt.subplots()
            if isinstance(shap_values, list):
                idx_class = 1 if len(shap_values) > 1 else 0
                display_shap = shap_values[idx_class]
            else:
                display_shap = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

            shap.summary_plot(display_shap, shap_test_data, show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.error(f"SHAP Error: {e}")

        # ================= INPUT MANUAL PASIEN BARU =================
        st.divider()
        st.header("🩺 Input Data Pasien Baru")
        st.write("Masukkan nilai manual untuk mendapatkan prediksi kondisi pasien berdasarkan model terbaik.")

        with st.form("form_prediksi"):
            col1, col2 = st.columns(2)
            user_input = {}
            
            # Loop hanya fitur yang terpilih oleh ANOVA
            for i, feat in enumerate(selected_features):
                container = col1 if i % 2 == 0 else col2
                
                # Cek jika fitur asli adalah numerik atau kategori
                if feat in num_cols:
                    # Ambil range dari data asli untuk bantuan user
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    user_input[feat] = container.number_input(f"{feat}", min_value=min_val, max_value=max_val, value=mean_val)
                else:
                    unique_vals = df[feat].unique().tolist()
                    user_input[feat] = container.selectbox(f"{feat}", unique_vals)
            
            btn_predict = st.form_submit_button("Analisis Kondisi Pasien")

        if btn_predict:
            # Buat DataFrame dari input
            input_df = pd.DataFrame([user_input])

            # Preprocessing Input: Label Encoding untuk Kategorikal
            for col in cat_cols:
                if col in input_df.columns:
                    # Gunakan encoder baru untuk input (fit pada data asli agar konsisten)
                    tmp_le = LabelEncoder().fit(df[col].astype(str))
                    input_df[col] = tmp_le.transform(input_df[col].astype(str))

            # Preprocessing Input: Scaling untuk Numerik
            if len(num_cols) > 0:
                cols_to_scale = [c for c in num_cols if c in input_df.columns]
                if cols_to_scale:
                    # Re-scale menggunakan scaler yang sudah ada
                    # Catatan: scaler butuh semua kolom numerik awal, kita isi sisanya dengan 0 sementara
                    full_input_scale = pd.DataFrame(0, index=[0], columns=num_cols)
                    for c in cols_to_scale:
                        full_input_scale[c] = input_df[c]
                    
                    scaled_vals = scaler.transform(full_input_scale)
                    # Kembalikan nilai yang sudah di-scale ke input_df
                    for i, c in enumerate(num_cols):
                        if c in input_df.columns:
                            input_df[c] = scaled_vals[0][i]

            # Prediksi
            final_pred = svm_opt.predict(input_df[selected_features])
            final_proba = svm_opt.predict_proba(input_df[selected_features])
            final_label = le_target.inverse_transform(final_pred)[0]

            # Tampilkan Hasil
            st.subheader("Hasil Diagnosis:")
            if final_pred[0] == 1:
                st.error(f"Prediksi: **{final_label}**")
            else:
                st.success(f"Prediksi: **{final_label}**")
            
            # Tampilkan Probabilitas
            prob_df = pd.DataFrame(final_proba, columns=le_target.classes_)
            st.write("Tingkat Keyakinan Model:")
            st.dataframe(prob_df)

            import pickle

            # Simpan objek-objek penting
            model_data = {
                "model": svm_opt,
                "scaler": scaler,
                "le_target": le_target,
                "selected_features": selected_features.tolist(),
                "num_cols": num_cols.tolist(),
                "cat_cols": cat_cols.tolist()
            }

            with open("medical_model.pkl", "wb") as f:
                pickle.dump(model_data, f)