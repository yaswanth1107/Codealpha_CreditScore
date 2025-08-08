# 💳 Creditworthiness Prediction Project

This project predicts whether a person is likely to default on credit card payments based on their past financial data.  
It includes:
- A **Jupyter Notebook / Python script** for training and evaluating multiple ML models (Logistic Regression, Decision Tree, Random Forest).
- A **Streamlit web app** (`credit_app.py`) that allows users to input financial details and get predictions in real time.

---

## 📂 Project Structure
```
.
├── credit_app.py        # Streamlit web application
├── requirements.txt     # Required Python packages
├── README.md            # Project documentation
└── UCI_Credit_Card.csv  # Dataset (download separately)
```

---

## 📊 Dataset
The project uses the [UCI Credit Card dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

**Columns**:
- ID — Client ID
- Financial attributes (LIMIT_BAL, PAY_0, PAY_2, BILL_AMT1…)
- Target: `default.payment.next.month` (1 = default, 0 = no default)

---

## ⚙️ Installation

1️⃣ **Clone or Download the Project**
```bash
git clone <your-repo-url>
cd <project-folder>
```

2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Add Dataset**
- Download `UCI_Credit_Card.csv`
- Place it in the project directory or update the file path in both scripts.

---

## 📈 Run the Analysis Script
To train and evaluate the models in your terminal:
```bash
python your_script.py
```
You will see:
- Classification Reports
- ROC AUC scores
- Confusion Matrices (visualized with Seaborn)

---

## 🌐 Run the Streamlit App
```bash
streamlit run credit_app.py
```
Then open the local URL provided by Streamlit in your browser.

---

## 🖼 Features of the Streamlit App
- Custom background image styling.
- Real-time input of financial details.
- Prediction output with **creditworthiness score**.
- Uses **Random Forest Classifier** trained on the dataset.

---

## 📌 Example Prediction
If the model predicts `1`:
```
❌ This person is likely to default! (Risk Score: 0.78)
```
If the model predicts `0`:
```
✅ This person is likely creditworthy! (Risk Score: 0.15)
```

---

## 📜 License
This project is for educational purposes only. No warranties are provided.
