import streamlit as st
import pandas as pd
import pyodbc

st.set_page_config(
    page_title='Database',
    page_icon='',
    layout='wide'
)

st.title('Customer churn data')

# Dictionary to store column descriptions
column_descriptions = {
    'gender': 'Gender of the customer',
    'SeniorCitizen': 'Whether the customer is a senior citizen (1) or not (0)',
    'Partner': 'Whether the customer has a partner (Yes) or not (No)',
    'Dependents': 'Whether the customer has dependents (Yes) or not (No)',
    'tenure': 'Number of months the customer has stayed with the company',
    'PhoneService': 'Whether the customer has phone service (Yes) or not (No)',
    'MultipleLines': 'Whether the customer has multiple lines (Yes, No, or No phone service)',
    'InternetService': 'Type of internet service the customer has (DSL, Fiber optic, or No)',
    'OnlineSecurity': 'Whether the customer has online security (Yes, No, or No internet service)',
    'OnlineBackup': 'Whether the customer has online backup (Yes, No, or No internet service)',
    'DeviceProtection': 'Whether the customer has device protection (Yes, No, or No internet service)',
    'TechSupport': 'Whether the customer has tech support (Yes, No, or No internet service)',
    'StreamingTV': 'Whether the customer has streaming TV (Yes, No, or No internet service)',
    'StreamingMovies': 'Whether the customer has streaming movies (Yes, No, or No internet service)',
    'Contract': 'Type of contract the customer has (Month-to-month, One year, Two year)',
    'PaperlessBilling': 'Whether the customer has paperless billing (Yes) or not (No)',
    'PaymentMethod': 'Payment method used by the customer (Electronic check, Mailed check, Bank transfer, Credit card)',
    'MonthlyCharges': 'Monthly charges of the customer',
    'TotalCharges': 'Total charges paid by the customer',
    'Churn': 'Whether the customer churned (Yes) or not (No)'
}

st.cache(allow_output_mutation=True, show_spinner=False)
def sourcing_data():
    connection = pyodbc.connect(
        "DRIVER={SQL Server};SERVER="
        + st.secrets["SERVER"]
        + ";DATABASE="
        + st.secrets["DATABASE"]
        + ";UID="
        + st.secrets["USER"]
        + ";PWD="
        + st.secrets["PASSWORD"]
    )
    return connection

def database_query(connection, query):
    with connection.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        df = pd.DataFrame.from_records(data=rows, columns=[column[0] for column in cur.description])
    connection.close()
    return df

if __name__ == "__main__":
    conn = sourcing_data()
    query = "SELECT * FROM LP2_Telco_churn_first_3000"
    df = database_query(conn, query)

    # Add a dropdown to select column type
    column_type = st.selectbox("Select Column Type", ["Categorical Columns", "Numerical Columns"])

    if column_type == "Categorical Columns":
        # Display columns with categorical values
        categorical_columns = df.select_dtypes(include=["object","bool"]).columns
        st.write("Categorical Columns:", categorical_columns)

    elif column_type == "Numerical Columns":
        # Display columns with numerical values
        numerical_columns = df.select_dtypes(exclude=["object","bool"]).columns
        st.write("Numerical Columns:", numerical_columns)

    else:
        st.write("Please select a column type.")

    # Display both columns and rows based on the selected column type
    st.subheader("Selected Data")
    if column_type == "Categorical Columns":
        selected_data = df[categorical_columns]
    elif column_type == "Numerical Columns":
        selected_data = df[numerical_columns]
    else:
        selected_data = df

    st.write(selected_data)

    # Display column descriptions
    st.subheader("Column Descriptions")
    for col in selected_data.columns:
        if col in column_descriptions:
            st.write(f"**{col}:** {column_descriptions[col]}")



