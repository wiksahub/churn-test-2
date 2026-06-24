# 🛒 E-Commerce Churn Predictor

Aplikasi Streamlit untuk memprediksi pelanggan e-commerce yang berpotensi churn.

## 🤖 Info Model

| Komponen       | Detail                        |
|----------------|-------------------------------|
| Model          | Gradient Boosting Classifier  |
| Preprocessor   | RobustScaler + OneHotEncoder  |
| Imbalance      | SMOTE                         |
| Fitur Input    | 18 fitur                      |
| Output         | Churn (1) / Tidak Churn (0)   |

## 📊 Fitur yang Digunakan

**Numerik:** Tenure, CityTier, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount

**Kategorikal:** PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus
