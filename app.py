import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Customer Churn E-Commerce",
    page_icon="🛒",
    layout="wide",
)

# ----------------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_ecommerce_churn_model.joblib")

model = load_model()

# ----------------------------------------------------------------------------
# SIDEBAR - NAVIGASI & BUSINESS CONTEXT
# ----------------------------------------------------------------------------
st.sidebar.title("🛒 Churn Prediction App")
page = st.sidebar.radio(
    "Navigasi",
    ["📊 Business Context", "🔮 Prediksi Churn", "📁 Prediksi Massal (Batch)"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Model: **Gradient Boosting Classifier**\n\n"
    "Pipeline: RobustScaler + One-Hot Encoding + SMOTE\n\n"
    "Dataset: E-Commerce Customer Churn"
)

# ----------------------------------------------------------------------------
# HALAMAN 1 - BUSINESS CONTEXT
# ----------------------------------------------------------------------------
if page == "📊 Business Context":
    st.title("📊 Business Context: Customer Churn Prediction")

    st.markdown("""
    ### Mengapa Prediksi Churn Itu Penting?

    Dalam bisnis **e-commerce**, **customer churn** terjadi ketika pelanggan berhenti
    berbelanja atau menggunakan platform dalam jangka waktu tertentu. Mempertahankan
    pelanggan yang sudah ada jauh lebih murah dibandingkan mendapatkan pelanggan baru
    — beberapa studi industri menyebutkan biaya akuisisi pelanggan baru bisa **5–25x**
    lebih mahal daripada biaya retensi.

    Dengan model **machine learning** ini, tim bisnis dapat:
    - 🎯 **Mengidentifikasi pelanggan berisiko churn** sebelum mereka benar-benar pergi
    - 💰 **Mengoptimalkan budget retensi** dengan menargetkan pelanggan yang tepat
    - 📈 **Meningkatkan customer lifetime value (CLV)** melalui intervensi yang tepat waktu
    - 🧠 **Memahami faktor pendorong churn** untuk perbaikan produk/layanan
    """)

    st.markdown("---")
    st.subheader("⚙️ Bagaimana Model Ini Bekerja?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Algoritma:** Gradient Boosting Classifier

        **Pipeline pemrosesan data:**
        1. *RobustScaler* — menyamakan skala fitur numerik (tahan terhadap outlier)
        2. *One-Hot Encoding* — mengubah fitur kategorikal menjadi numerik
        3. *SMOTE* — menyeimbangkan data antara pelanggan churn & tidak churn saat training
        4. *Gradient Boosting* — menghasilkan prediksi probabilitas churn
        """)
    with col2:
        st.markdown("""
        **Output model:**
        - **Churn (1):** Pelanggan berisiko/diprediksi akan churn
        - **Tidak Churn (0):** Pelanggan diprediksi tetap loyal
        - Disertai **probabilitas/confidence score** untuk membantu prioritas tindak lanjut
        """)

    st.markdown("---")
    st.subheader("🔑 Faktor-Faktor Paling Berpengaruh Terhadap Churn")
    st.markdown("""
    Berdasarkan analisis *feature importance* dari model:

    | Faktor | Insight Bisnis |
    |---|---|
    | **Tenure** (lama berlangganan) | Pelanggan baru jauh lebih berisiko churn dibanding pelanggan lama → fokuskan program onboarding & engagement awal |
    | **Complain** (riwayat komplain) | Pelanggan yang pernah komplain memiliki risiko churn lebih tinggi → perkuat penanganan *customer service* |
    | **Number of Address** | Banyaknya alamat terdaftar berkorelasi dengan pola transaksi/perilaku pelanggan |
    | **Number of Devices Registered** | Mencerminkan tingkat keterlibatan pelanggan dengan platform |
    | **Satisfaction Score** | Skor kepuasan rendah → sinyal awal risiko churn |
    | **Cashback Amount & Day Since Last Order** | Insentif dan keaktifan transaksi terakhir memengaruhi loyalitas |

    **Rekomendasi aksi:** Pelanggan baru, pernah komplain, dan jarang bertransaksi
    sebaiknya menjadi prioritas utama kampanye retensi (diskon personal, follow-up
    customer service, program loyalitas).
    """)

    st.markdown("---")
    st.subheader("👉 Gunakan menu di sidebar")
    st.markdown("""
    - **🔮 Prediksi Churn** — untuk memprediksi 1 pelanggan secara interaktif
    - **📁 Prediksi Massal (Batch)** — untuk memprediksi banyak pelanggan sekaligus via upload CSV
    """)

# ----------------------------------------------------------------------------
# HALAMAN 2 - PREDIKSI INDIVIDUAL
# ----------------------------------------------------------------------------
elif page == "🔮 Prediksi Churn":
    st.title("🔮 Prediksi Churn Pelanggan")
    st.markdown("Masukkan data pelanggan di bawah ini untuk memprediksi kemungkinan churn.")

    with st.form("churn_form"):
        st.subheader("👤 Profil Pelanggan")
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.number_input("Tenure (bulan berlangganan)", min_value=0, max_value=100, value=12)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
            city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        with c3:
            preferred_login_device = st.selectbox("Perangkat Login Favorit", ["Mobile Phone", "Computer"])
            preferred_payment_mode = st.selectbox(
                "Metode Pembayaran Favorit",
                ["Debit Card", "Credit Card", "E wallet", "Cash on Delivery", "UPI"],
            )

        st.subheader("📱 Aktivitas & Engagement")
        c4, c5, c6 = st.columns(3)
        with c4:
            hour_spend_on_app = st.number_input("Jam yang dihabiskan di app/hari", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
            number_of_device_registered = st.number_input("Jumlah Device Terdaftar", min_value=1, max_value=10, value=3)
        with c5:
            satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
            number_of_address = st.number_input("Jumlah Alamat Terdaftar", min_value=1, max_value=20, value=2)
        with c6:
            complain = st.selectbox("Pernah Komplain?", ["Tidak", "Ya"])
            complain = 1 if complain == "Ya" else 0
            warehouse_to_home = st.number_input("Jarak Gudang ke Rumah (km)", min_value=0, max_value=200, value=15)

        st.subheader("🛍️ Riwayat Transaksi")
        c7, c8, c9 = st.columns(3)
        with c7:
            preferred_order_cat = st.selectbox(
                "Kategori Order Favorit",
                ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"],
            )
            order_count = st.number_input("Jumlah Order (bulan terakhir)", min_value=0, max_value=50, value=3)
        with c8:
            day_since_last_order = st.number_input("Hari Sejak Order Terakhir", min_value=0, max_value=365, value=5)
            coupon_used = st.number_input("Jumlah Kupon Digunakan", min_value=0, max_value=50, value=1)
        with c9:
            order_amount_hike = st.number_input("Kenaikan Order Amount dari Tahun Lalu (%)", min_value=0, max_value=100, value=15)
            cashback_amount = st.number_input("Cashback Amount (Rp ribuan)", min_value=0.0, max_value=1000.0, value=150.0)

        average_cashback = st.number_input("Rata-rata Cashback Historis (Rp ribuan)", min_value=0.0, max_value=1000.0, value=150.0)

        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Tenure": tenure,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_to_home,
            "HourSpendOnApp": hour_spend_on_app,
            "NumberOfDeviceRegistered": number_of_device_registered,
            "SatisfactionScore": satisfaction_score,
            "NumberOfAddress": number_of_address,
            "Complain": complain,
            "OrderAmountHikeFromlastYear": order_amount_hike,
            "CouponUsed": coupon_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": day_since_last_order,
            "CashbackAmount": cashback_amount,
            "AverageCashback": average_cashback,
            "PreferredLoginDevice": preferred_login_device,
            "PreferredPaymentMode": preferred_payment_mode,
            "Gender": gender,
            "PreferedOrderCat": preferred_order_cat,
            "MaritalStatus": marital_status,
        }])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        churn_proba = proba[1]

        st.markdown("---")
        st.subheader("📋 Hasil Prediksi")

        col1, col2 = st.columns([1, 2])
        with col1:
            if pred == 1:
                st.error("⚠️ **BERISIKO CHURN**")
            else:
                st.success("✅ **PELANGGAN LOYAL**")
            st.metric("Probabilitas Churn", f"{churn_proba*100:.1f}%")

        with col2:
            st.progress(float(churn_proba))
            if churn_proba >= 0.7:
                st.warning(
                    "🔴 **Risiko Tinggi.** Rekomendasi: hubungi pelanggan secara personal, "
                    "tawarkan promo retensi/cashback khusus, dan pastikan tidak ada keluhan yang belum terselesaikan."
                )
            elif churn_proba >= 0.4:
                st.warning(
                    "🟠 **Risiko Sedang.** Rekomendasi: kirim notifikasi/email engagement, "
                    "tawarkan kupon ringan, pantau aktivitas dalam 2-4 minggu ke depan."
                )
            else:
                st.info(
                    "🟢 **Risiko Rendah.** Pelanggan tampak loyal. Tetap jaga pengalaman "
                    "berbelanja dan pertimbangkan program loyalitas jangka panjang."
                )

        with st.expander("🔍 Lihat data input yang digunakan"):
            st.dataframe(input_df.T.rename(columns={0: "Nilai"}))

# ----------------------------------------------------------------------------
# HALAMAN 3 - PREDIKSI BATCH
# ----------------------------------------------------------------------------
else:
    st.title("📁 Prediksi Massal (Batch) via CSV")
    st.markdown("""
    Upload file CSV berisi data pelanggan untuk memprediksi churn secara massal.
    Pastikan kolom CSV sesuai dengan format berikut:
    """)

    required_cols = [
        "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
        "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
        "DaySinceLastOrder", "CashbackAmount", "AverageCashback",
        "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
        "PreferedOrderCat", "MaritalStatus",
    ]
    st.code(", ".join(required_cols))

    # Template CSV
    template_df = pd.DataFrame([{
        "Tenure": 12, "CityTier": 1, "WarehouseToHome": 15, "HourSpendOnApp": 3.0,
        "NumberOfDeviceRegistered": 3, "SatisfactionScore": 3, "NumberOfAddress": 2,
        "Complain": 0, "OrderAmountHikeFromlastYear": 15, "CouponUsed": 1, "OrderCount": 3,
        "DaySinceLastOrder": 5, "CashbackAmount": 150.0, "AverageCashback": 150.0,
        "PreferredLoginDevice": "Mobile Phone", "PreferredPaymentMode": "Debit Card",
        "Gender": "Female", "PreferedOrderCat": "Laptop & Accessory", "MaritalStatus": "Single",
    }])
    st.download_button(
        "⬇️ Download Template CSV",
        template_df.to_csv(index=False).encode("utf-8"),
        file_name="template_churn_prediction.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            missing_cols = [c for c in required_cols if c not in data.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan di file Anda: {missing_cols}")
            else:
                preds = model.predict(data[required_cols])
                probas = model.predict_proba(data[required_cols])[:, 1]

                result = data.copy()
                result["Churn_Prediction"] = np.where(preds == 1, "Churn", "Tidak Churn")
                result["Churn_Probability (%)"] = (probas * 100).round(1)

                st.success(f"✅ Prediksi berhasil untuk {len(result)} pelanggan.")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Pelanggan", len(result))
                col2.metric("Diprediksi Churn", int((preds == 1).sum()))
                col3.metric("Churn Rate", f"{(preds == 1).mean()*100:.1f}%")

                st.dataframe(result, use_container_width=True)

                st.download_button(
                    "⬇️ Download Hasil Prediksi (CSV)",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_churn.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca/memproses file: {e}")
