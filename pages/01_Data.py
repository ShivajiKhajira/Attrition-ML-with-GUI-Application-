
import streamlit as st
import pandas as pd
import pyodbc

st.set_page_config(
    page_title='Database',
    page_icon='',
    layout='wide'
)

st.title('Customer churn data')

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



