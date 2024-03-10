import joblib
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd


 # Set the title of the app
st.title("Customer Churn Prediction Page")

# st.cache_resource(show_spinner='Model loading')
def load_decision_tree_model():
    pipeline = joblib.load('Models/decision_tree_model.pkl')
    return pipeline

# st.cache_resource(show_spinner='Model loading')
def load_random_forest_model():
    pipeline = joblib.load('Models/random_forest_model.pkl')
    return pipeline
# st.cache_resource()

def prediction_model():
    st.selectbox("Select preferred Prediction Model", options= ['Decision Tree', 'Random Forest'], key='Selected Model')
    
    if st.session_state['Selected Model'] == 'Decision Tree' :
        pipeline = load_decision_tree_model()
    else:
        pipeline = load_random_forest_model()
    encoder = joblib.load('Models/label_encoder.pkl')
    return pipeline, encoder

def machine_prediction (df, pipeline, encoder):

    pred = pipeline.predict(df)

    prediction = int(pred[0])
    prediction = encoder.inverse_transform([prediction])
    # st.session_state['Customer Churn'] = prediction

    probability =pipeline.predict_proba(df)

    # st.session_state['prediction'] = prediction
    # st.session_state['probability'] = probability
    return prediction, probability

# if 'prediction' not in st.session_state:
#     st.session_state['prediction'] = None

def inputs(pipeline,encoder):
   with st.form('form-input'):
    # Category 1: Demographic Information
    st.header("Demographic Information")
    gender, senior_citizen, partner, dependents = st.columns(4)
    gender = gender.selectbox("Gender", ('Male', 'Female'), key='gender')
    senior_citizen = senior_citizen.selectbox("Senior Citizen", ('Yes', 'No'), key='senior_citizen')
    partner = partner.selectbox("Partner", ('Yes', 'No'), key='partner')
    dependents = dependents.selectbox("Dependents", ('Yes', 'No'), key='dependents')

    # Category 2: Customer Tenure
    st.header("Customer Tenure")
    tenure = st.slider("Tenure (months)", 0, 81, 40, key='tenure')

    # Category 3: Service Subscriptions
    st.header("Service Subscriptions")
    service_cols = st.columns(3)
    with service_cols[0]:
        phone_service = st.selectbox("Phone Service", ('Yes', 'No'), key='phone_service')
        multiple_lines = st.selectbox("Multiple Lines", ('Yes', 'No'), key='multiple_lines')
        internet_service = st.selectbox("Internet Service", ('DSL', 'Fibre Optic', 'No'), key='internet_service')
    with service_cols[1]:
        online_security = st.selectbox("Online Security", ('Yes', 'No'), key='online_security')
        online_backup = st.selectbox("Online Backup", ('Yes', 'No'), key='online_backup')
        device_protection = st.selectbox("Device Protection", ('Yes', 'No'), key='device_protection')
    with service_cols[2]:
        tech_support = st.selectbox("Tech Support", ('Yes', 'No'), key='tech_support')
        streaming_tv = st.selectbox("Streaming TV", ('Yes', 'No'), key='streaming_tv')
        streaming_movies = st.selectbox("Streaming Movies", ('Yes', 'No'), key='streaming_movies')

    # Category 4: Contract and Billing
    st.header("Contract and Billing")
    billing_cols = st.columns(3)
    with billing_cols[0]:
        contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'), key='contract')
        paperless_billing = st.selectbox("Paperless Billing", ('Yes', 'No'), key='paperless_billing')
    with billing_cols[1]:
        payment_method = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'), key='payment_method')

    # Category 5: Financial Information
    st.header("Financial Information")
    financial_cols = st.columns(2)
    with financial_cols[0]:
        monthly_charges = st.slider("Monthly Charges", 0, 5000, 100, key='monthly_charges')
    with financial_cols[1]:
        total_charges = st.slider("Total Charges", 0, 20000, 5000, key='total_charges')

    submit_button = st.form_submit_button('Submit Inputs')

    if submit_button:
        gender = st.session_state['gender'] 
        senior_citizen = st.session_state['senior_citizen']
        partner = st.session_state['partner'] 
        dependents = st.session_state['dependents'] 
        phone_service = st.session_state['phone_service'] 
        multiple_lines = st.session_state['multiple_lines'] 
        internet_service = st.session_state['internet_service']
        online_security  = st.session_state['online_security']
        online_backup  = st.session_state['online_backup']
        device_protection  = st.session_state['device_protection']
        tech_support  = st.session_state['tech_support']
        streaming_tv  = st.session_state['streaming_tv']
        streaming_movies = st.session_state['streaming_movies']
        contract = st.session_state['contract']
        paperless_billing = st.session_state['paperless_billing']
        payment_method = st.session_state['payment_method']
        monthly_charges = st.session_state['monthly_charges']
        total_charges = st.session_state['total_charges']

        columns = ['gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'internet_service', 
        'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
        'streaming_movies', 'contract', 'paperless_billing', 'payment_method', 'monthly_charges', 'total_charge']

        data = [[gender, senior_citizen, partner, dependents, phone_service, multiple_lines, internet_service, 
        online_security, online_backup, device_protection, tech_support, streaming_tv, 
        streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges]]

        df = pd.DataFrame(data, columns=columns)
        # st.form_submit_button('Submit Inputs', on_click=machine_prediction, kwargs= dict(df=df,pipeline=pipeline, encoder=encoder))
        
        return df, pipeline, encoder





if __name__ == "__main__":
    # Load your pipeline and encoder
    # Replace the following with your actual pipeline and encoder initialization
    pipeline, encoder = prediction_model()

    # Your input form function (you can adapt this based on your input method)
    df, pipeline, encoder = inputs(pipeline, encoder)

    st.write(df)
    # st.write(pipeline)
    st.write(encoder)

    # Check if 'prediction' is not in st.session_state
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    
    if 'probability' not in st.session_state:
        st.session_state['probability'] = None

    # Call the machine_prediction function
    prediction, probability = machine_prediction(df, pipeline, encoder)

    # Update session_state with the results
    st.session_state['prediction'] = prediction
    st.session_state['probability'] = probability

    # Display the result
    st.write("Prediction:", prediction)
    st.write("Probability:", probability)



    # pipeline, encoder = prediction_model()
    # st.write(pipeline)
    # st.write(encoder)
    # outputs = inputs(pipeline,encoder)
    # st.write(outputs)
#     """ final_output= st.session_state['']
#     probability = st.session_state['probability']

#     if not final_output:
#         st.markdown("###Prediction will show here")
#     elif final_output =="Yes":
#         churn_probability = probability[0][1] * 100
#         st.markdown("### The customer will churn with a probability of {round(churn_probability), 2}%")

#     else:
#         probability_of_no = probability[0][0] * 100
#         st.markdown("### Customer will not churn with a probability of {round(probability_of_no, 2)}%")
#     st.markdown(f" ###{final_output}")
#     st.write(st.session_state)

#  """