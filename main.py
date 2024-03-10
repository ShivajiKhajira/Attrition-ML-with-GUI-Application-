import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title='Home',
    page_icon=':)',
    layout='wide'
)
st.markdown(
    """
    <style>
        body {
            background-image: url('https://www.pinterest.com/pin/739857045018933408/');
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def authenticate(username, password):
    # Replace this with your authentication logic
    return username == "admin" and password == "password"

def main():
    # Use st.columns to create two columns
    col1, col2 = st.columns(2)

    # Set the title of the app
    col1.title("Secure Data App")

    # Main Login Form
    username = col2.text_input("Username")
    password = col2.text_input("Password", type="password")

    if col2.button("Login"):
        if authenticate(username, password):
            col2.success("Logged in as {}".format(username))

            # Add your main application content here
            st.write("Welcome to the secure data app!")

        else:
            col2.error("Invalid credentials")

if __name__ == "__main__":
    main()