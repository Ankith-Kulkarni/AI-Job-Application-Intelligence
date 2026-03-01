import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score


st.set_page_config(
    page_title="AI Job Application Intelligence",
    layout="wide",
    page_icon="📊"
)

st.title("📊 AI-Powered Job Application Intelligence System")
st.markdown("Smart tracking + ML-based outcome prediction")

st.markdown("""
<style>
    .stMetric {
        background-color: #f5f7fa;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_connection():
    return sqlite3.connect("Applications.db", check_same_thread=False)

conn = get_connection()
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS companies ( 
    Com_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    Company_name TEXT, 
    Area TEXT, 
    City TEXT,   
    Status TEXT, 
    Roles TEXT, 
    Package REAL, 
    Experience REAL, 
    Work_mode TEXT, 
    Skills TEXT, 
    Application_date TEXT, 
    Follow_up1 TEXT, 
    Follow_up2 TEXT, 
    Follow_up3 TEXT, 
    Shift TEXT
)
""")
conn.commit()

def load_data():
    return pd.read_sql("SELECT * FROM companies", conn)

def feature_engineering(df):
    df["Package"] = pd.to_numeric(df["Package"], errors="coerce")
    df["Experience"] = pd.to_numeric(df["Experience"], errors="coerce")
    df["Application_date"] = pd.to_datetime(df["Application_date"], errors="coerce")
    df["Application_date"] = df["Application_date"].fillna(datetime.now())
    df["Application_month"] = df["Application_date"].dt.month
    df = df.fillna(0)
    return df

def train_model(df):
    df = feature_engineering(df)

    y = df["Status"]
    X = df.drop(columns=[
        "Com_id", "Status",
        "Skills", "Application_date",
        "Follow_up1", "Follow_up2", "Follow_up3"
    ], errors="ignore")

    categorical_cols = ["Company_name", "Area", "City", "Roles", "Work_mode", "Shift"]
    numeric_cols = ["Package", "Experience", "Application_month"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=y if y.value_counts().min() > 1 else None
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    joblib.dump(model, "model.pkl")
    return model, metrics

menu = st.sidebar.radio(
    "Navigation",
    ["Add Application", "Dashboard", "Upload File",
     "Delete Applications", "Insights", "Predict Outcome"]
)

if menu == "Add Application":
    st.header("➕ Add New Application")

    with st.form("add_form"):
        col1, col2 = st.columns(2)

        with col1:
            company_name = st.text_input("Company Name", "")
            area = st.text_input("Area", "")
            city = st.text_input("City", "")
            roles = st.text_input("Role", "")
            status = st.selectbox("Status",
                                  ["Applied", "Interview", "Offer",
                                   "Rejected", "Pending", "Shortlisted"])

        with col2:
            package = st.number_input("Package (CTC)", 0.0)
            experience = st.number_input("Experience (Years)", 0.0)
            work_mode = st.selectbox("Work Mode", ["WFH", "WFO", "Hybrid"])
            shift = st.selectbox("Shift", ["Day", "Night", "Flexible"])
            application_date = st.date_input("Application Date")

        submitted = st.form_submit_button("Save Application")

        if submitted:
            if company_name.strip() == "" or roles.strip() == "":
                st.error("⚠️ Company Name and Role are required.")
            else:
                c.execute("""
                    INSERT INTO companies (
                        Company_name, Area, City,
                        Status, Roles, Package, Experience, Work_mode, Skills,
                        Application_date, Follow_up1, Follow_up2, Follow_up3, Shift
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    company_name, area, city,
                    status, roles, package, experience,
                    work_mode, "", str(application_date),
                    "", "", "", shift
                ))
                conn.commit()
                st.success("✅ Application Saved Successfully!")

elif menu == "Dashboard":
    st.header("📋 Application Overview")

    df = load_data()

    if df.empty:
        st.warning("⚠️ No applications found.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Applications", len(df))
        col2.metric("Offers", (df["Status"] == "Offer").sum())
        col3.metric("Interviews", (df["Status"] == "Interview").sum())
        col4.metric("Rejected", (df["Status"] == "Rejected").sum())

        st.subheader("🔍 Filters")
        company = st.selectbox("Company", ["All"] + sorted(df["Company_name"].dropna().unique()))
        status = st.selectbox("Status", ["All"] + sorted(df["Status"].dropna().unique()))

        if company != "All":
            df = df[df["Company_name"] == company]
        if status != "All":
            df = df[df["Status"] == status]

        df_display = df.sort_values("Application_date", ascending=False).reset_index(drop=True)
        st.dataframe(df_display, use_container_width=True)

elif menu == "Insights":
    st.header("📊 Application Success Analytics")

    df = load_data()
    df = feature_engineering(df)

    if not df.empty:
        st.subheader("Success Rate by City")
        success_city = (
            df[df["Status"] == "Offer"]
            .groupby("City")
            .size() /
            df.groupby("City").size()
        )
        st.bar_chart(success_city)

        st.subheader("Success Rate by Work Mode")
        success_mode = (
            df[df["Status"] == "Offer"]
            .groupby("Work_mode")
            .size() /
            df.groupby("Work_mode").size()
        )
        st.bar_chart(success_mode)
    else:
        st.warning("⚠️ No data available for insights.")


if menu == "Upload File":
    st.header("📂 Upload Applications File")

    expected_cols = [
        "Com_id", "Company_name", "Area", "City",
        "Status", "Roles", "Package", "Experience", "Work_mode", "Skills",
        "Application_date", "Follow_up1", "Follow_up2", "Follow_up3", "Shift"
    ]

    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if file:
        try:
            if file.name.endswith(".csv"):
                # FIX: removed 'errors' argument
                df_upload = pd.read_csv(file, encoding="cp1252")
            else:
                df_upload = pd.read_excel(file)

            uploaded_cols = list(df_upload.columns)
            if set(expected_cols) != set(uploaded_cols):
                st.warning("⚠️ Column headers in the uploaded file do not match the required schema. "
                           "Please ensure your file has the following columns:\n\n" + ", ".join(expected_cols))
            else:
                st.dataframe(df_upload, use_container_width=True)

                if st.button("Save to Database"):
                    df_upload.to_sql("companies", conn, if_exists="append", index=False)
                    st.success("✅ Data Uploaded Successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")


elif menu == "Delete Applications":
    st.header("🗑️ Delete Application")

    df = load_data()
    delete_id = st.selectbox("Select Company", ["All"] + df["Company_name"].tolist())

    if st.button("Delete"):
        if delete_id == "All":
            c.execute("DELETE FROM companies")
        else:
            c.execute("DELETE FROM companies WHERE Company_name = ?", (delete_id,))
        conn.commit()
        st.success("✅ Deleted Successfully")

elif menu == "Predict Outcome":
    st.header("🔮 ML-Based Outcome Prediction")

    df = load_data()

    if df.empty or df["Status"].nunique() < 2:
        st.warning("⚠️ Not enough labeled data.")
        st.stop()

    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
    else:
        model, metrics = train_model(df)
        st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        st.metric("F1 Score", f"{metrics['f1']:.2f}")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            company_name = st.text_input("Company Name")
            area = st.text_input("Area")
            city = st.text_input("City")
            roles = st.text_input("Role")

        with col2:
            package = st.number_input("Package", 0.0)
            experience = st.number_input("Experience", 0.0)
            work_mode = st.selectbox("Work Mode", ["WFH", "WFO", "Hybrid"])
            shift = st.selectbox("Shift", ["Day", "Night", "Flexible"])

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        new_df = pd.DataFrame([{
            "Company_name": company_name,
            "Area": area,
            "City": city,
            "Roles": roles,
            "Package": package,
            "Experience": experience,
            "Work_mode": work_mode,
            "Shift": shift,
            "Application_month": datetime.now().month
        }])

        prediction = model.predict(new_df)[0]
        probabilities = model.predict_proba(new_df)[0]

        st.success(f"🧠 Predicted Outcome: {prediction}")

        prob_df = pd.DataFrame({
            "Status": model.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        prob_df = prob_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True)

        st.bar_chart(prob_df.set_index("Status"))

        st.metric("Model Confidence", f"{max(probabilities):.2%}")












