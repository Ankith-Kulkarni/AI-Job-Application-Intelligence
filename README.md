# 📊 AI Job Application Intelligence System

*An end-to-end **AI-powered job application tracking system** that helps users analyze application performance, uncover success patterns, and predict outcomes using **machine learning**.* 

This project combines **SQLite**, **Pandas**, and **Scikit-learn** with an interactive UI for smarter career management.

---

## 🚀 Features

- ➕ **Add Applications**: Store company, role, package, experience, and status.
- 📋 **Dashboard**: View total applications, offers, interviews, and rejections with filters.
- 📂 **Upload Files**: Import applications from CSV/Excel with schema validation.
- 🗑️ **Delete Applications**: Remove single or all records easily.
- 📊 **Insights**: Success analytics by city and work mode.
- 🔮 **Predict Outcome**: ML-powered prediction of application results with confidence scores.

---

## 🏗 System Architecture
```text
User Input (Streamlit UI)
        ↓
SQLite Database
        ↓
Pandas Data Processing
        ↓
Scikit-learn ML Pipeline
        ↓
Random Forest Model
        ↓
Prediction Output with Confidence Score
```

## Machine Learning Model

**Model Used:** *RandomForestClassifier*

**Target Variable:** *Application Status (Offer / Interview / Rejected)*

**Features Used:**
- City
- Work Mode
- Experience Required
- Package (LPA)
- Role

**Preprocessing Pipeline:**
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- ColumnTransformer for feature management
- Integrated Pipeline for reproducibility

**Evaluation Metrics:**
- Accuracy Score
- F1-Score
- Classification Report

*The model enables predictive analysis to identify high-probability job applications.*


## Dataset Schema
| Column Name      | Description                            |
| ---------------- | -------------------------------------- |
| Company          | Company Name                           |
| Role             | Job Title                              |
| City             | Job Location                           |
| Work_Mode        | Remote / Hybrid / Onsite               |
| Package_LPA      | Salary Offered (LPA)                   |
| Experience_Years | Required Experience                    |
| Status           | Applied / Interview / Offer / Rejected |
| Application_Date | Date of Application                    |


## 🛠 Tech Stack

- **Frontend/UI**: [Streamlit]
- **Database**: [SQLite]
- **Data Handling**: [Pandas]
- **Machine Learning**: [Scikit-learn] [RandomForestClassifier]
- **Model Persistence**: [Joblib]

---


## ▶️ Run Locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/Ankith-Kulkarni/AI-Job-Application-Intelligence.git
cd AI-Job-Application-Intelligence
pip install -r requirements.txt
streamlit run app.py
```

---

## 🖼️ Demo

Here are some screenshots of the dashboard in action:

### Dashboard Overview
*Displays overall performance metrics and application tracking summary.*

![Dashboard](docs/dashboard.png)

### Insights
*City-wise and work mode success analysis with interactive filters.*

![Insights](docs/analytics.png)

### Prediction Results
*Predicts job outcome probability with confidence score visualization.*

![Prediction Screenshot](docs/prediction.png)


## 👨‍💻 Author

**Ankith Kulkarni**  
B.Com (Honours)  
Certified Data Analyst  
Aspiring Data Scientist  

📫 Connect with me on LinkedIn:  
[Ankith Kulkarni](https://www.linkedin.com/in/ankith-kulkarni-64310422a)



