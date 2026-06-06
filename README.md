# 🌊 Telangana Flood Impact Prediction System

A Flask-based web application that predicts agricultural crop damage and economic losses caused by flood events across all 33 districts of Telangana, India.

> **Transparency Note:** This project was developed with AI assistance (Claude by Anthropic). The dataset includes agricultural records supplemented with synthetic data generated using ICAR crop parameters and IMD rainfall thresholds. Predictions are indicative estimates for academic purposes and should **not** be used for official disaster planning.

---

## 📸 Features

- 🌾 **Crop Damage Prediction** — Estimates percentage crop loss for 17 Telangana crops using a Random Forest model and ICAR agronomic damage curves
- 💸 **Economic Loss Analysis** — Calculates direct loss, supply chain impact, and total economic impact using sector multipliers
- 🗺️ **District-wise Risk Profiling** — Zone-based vulnerability scores for all 33 Telangana districts
- 📊 **Analytics Dashboard** — Model performance metrics (MAE, RMSE, R², Cross-validation)
- 🌡️ **Scenario Analysis** — Compare baseline vs moderate vs extreme flood scenarios
- 📋 **Actionable Recommendations** — Evacuation alerts, insurance guidance, market warnings

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | scikit-learn (Random Forest Regressor) |
| Data Processing | pandas, NumPy |
| Frontend | HTML, CSS, Jinja2 templates |
| Deployment | Render.com |

---

## 📁 Project Structure

```
telangana-flood-prediction/
├── app.py
├── requirements.txt
├── README.md
├── static/
│   └── images/
│       ├── screenshots/
│       │   
│       └── docs/
│
└── templates/
    ├── about.html                  # Project overview page
    ├── index.html                  # Prediction form
    ├── result.html                 # Prediction results
    ├── analytics.html              # Analytics dashboard
    ├── scenario.html               # Scenario analysis
    ├── data.html                   # Dataset viewer
    ├── 404.html                    # Error page
    └── 500.html                    # Error page
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Predicting-Flood-Impact-On-Agriculture-And-Economy.git
cd Predicting-Flood-Impact-On-Agriculture-And-Economy
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

> If `telangana_floods_official.csv` is not present, the app automatically generates a sample dataset of 50 records and continues running.

---

## 🤖 ML Model Details

| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Features | 15 (rainfall, duration, elevation, soil type, crop stage, etc.) |
| Target | Crop loss percentage |
| Train/Test Split | 80% / 20% |
| Cross-validation | 5-fold |

Model performance varies depending on the dataset. When run on the auto-generated sample dataset, metrics reflect that limited data.

---

## 🌾 Supported Crops

Rice, Cotton, Maize, Soybean, Chillies, Red Gram, Green Gram, Black Gram, Sugarcane, Groundnut, Castor, Jowar, Turmeric, Vegetables, Sweet Orange, Mango

---

## 🗺️ Districts Covered

All 33 districts of Telangana, including Hyderabad, Warangal, Khammam, Nalgonda, Karimnagar, and more — categorised into North, Central, and South agro-climatic zones.

---

## 🚀 Deployment (Render.com)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3

---

## ⚠️ Disclaimer

This system is an academic project built for a B.Tech Data Science final year submission. Predictions are based on a partially synthetic dataset and a simplified ML model. They are **not** a substitute for official government flood assessments or NDMA/SDMA advisories.

---

## 👤 Author

** Meer Adeen Ali **  
B.Tech Data Science — Sphoorthy Engineering College  
📧 [meeradeenali01@gmail.com]  
🔗 [LinkedIn] | [GitHub]

---

## 📄 License

This project is for academic and portfolio purposes. Feel free to fork and adapt with attribution.
