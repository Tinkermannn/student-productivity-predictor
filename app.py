import streamlit as st
import pandas as pd
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Prediktor Produktivitas Mahasiswa",
    layout="centered"
)

# Judul
st.title("Prediktor Produktivitas Mahasiswa")

st.markdown("""
Aplikasi ini memprediksi skor produktivitas mahasiswa berdasarkan
pola belajar, gaya hidup, dan kebiasaan akademik menggunakan
model Linear Regression.
""")

st.divider()

# Load model
@st.cache_resource
def load_models():
    try:
        required_files = [
            'scaler_final.pkl',
            'linear_regression_final.pkl',
            'selected_features.pkl'
        ]

        for file in required_files:
            if not os.path.exists(file):
                st.error(f"File {file} tidak ditemukan.")
                return None, None, None

        scaler = joblib.load('scaler_final.pkl')
        selected_features = joblib.load('selected_features.pkl')
        model = joblib.load('linear_regression_final.pkl')

        return scaler, selected_features, model

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None


scaler, selected_features, model = load_models()

# Form input
if model is not None:

    st.subheader("Input Data Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Akademik")

        study_hours = st.number_input(
            "Waktu Belajar per Hari",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        attendance = st.slider(
            "Persentase Kehadiran",
            min_value=0.0,
            max_value=100.0,
            value=85.0
        )

        focus_score = st.slider(
            "Skor Fokus",
            min_value=1,
            max_value=10,
            value=7
        )

    with col2:

        st.markdown("### Gaya Hidup")

        sleep_hours = st.number_input(
            "Waktu Tidur per Hari",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

        phone_usage = st.number_input(
            "Waktu Main HP per Hari",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

        stress_level = st.slider(
            "Tingkat Stres",
            min_value=1,
            max_value=10,
            value=5
        )

    st.markdown("")

    if st.button("Prediksi", use_container_width=True):

        # Input data
        input_data = {
            'study_hours_per_day': [study_hours],
            'sleep_hours': [sleep_hours],
            'phone_usage_hours': [phone_usage],
            'attendance_percentage': [attendance],
            'stress_level': [stress_level],
            'focus_score': [focus_score]
        }

        input_df = pd.DataFrame(input_data)

        try:
            # Samakan urutan dan nama kolom dengan scaler
            expected_columns = scaler.feature_names_in_

            # Debug info
            st.write("Kolom model:", expected_columns.tolist())
            st.write("Kolom input:", input_df.columns.tolist())

            # Validasi kolom
            missing_cols = [
                col for col in expected_columns
                if col not in input_df.columns
            ]

            if missing_cols:
                st.error(f"Kolom tidak ditemukan: {missing_cols}")
                st.stop()

            # Reorder kolom sesuai training
            input_df = input_df[expected_columns]

            # Scaling
            input_scaled = scaler.transform(input_df)

            # Prediksi
            prediction = model.predict(input_scaled)[0]

            # Batasi range
            prediction = max(0, min(100, prediction))

            st.divider()

            st.subheader("Hasil Prediksi")

            st.metric(
                label="Skor Produktivitas",
                value=f"{prediction:.2f} / 100"
            )

            # Interpretasi hasil
            if prediction >= 80:
                st.success("Produktivitas sangat baik.")

            elif prediction >= 60:
                st.info("Produktivitas cukup baik.")

            elif prediction >= 40:
                st.warning("Produktivitas sedang.")

            else:
                st.error("Produktivitas rendah.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.markdown("---")
st.caption("Kelompok 7 - Dasar Analitika Data")