import streamlit as st
import joblib
import pandas as pd

# Load the pipeline and categorical columns
@st.cache_resource
def load_model():
    model = joblib.load("financial_inclusion_model.pkl")
    categorical_cols = joblib.load("categorical_columns.pkl")
    return model, categorical_cols

model, categorical_cols = load_model()

# Category choices
choices = {
    'country': ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'],
    'location_type': ['Rural', 'Urban'],
    'cellphone_access': ['No', 'Yes'],
    'gender_of_respondent': ['Male', 'Female'],
    'relationship_with_head': ['Child', 'Head of Household', 'Other non-relatives', 
                              'Other relative', 'Parent', 'Spouse'],
    'marital_status': ['Divorced/Seperated', 'Married/Living together', 
                       'Single/Never Married', 'Widowed'],
    'education_level': ['No formal education', 'Primary education', 
                        'Secondary education', 'Tertiary education', 
                        'Vocational/Specialised training'],
    'job_type': ['Farming and Fishing', 'Formally employed Government',
                 'Formally employed Private', 'Government Dependent', 
                 'Informally employed', 'Other Income',
                 'Remittance Dependent', 'Self employed']
}

st.title("📊 Financial Inclusion Predictor")
st.header("📝 Enter Respondent Information")

# Input form
with st.form("predict_form"):
    input_data = {
        'country': st.selectbox('Country', choices['country']),
        'location_type': st.selectbox('Location Type', choices['location_type']),
        'cellphone_access': st.selectbox('Cellphone Access', choices['cellphone_access']),
        'household_size': st.number_input('Household Size', min_value=1, max_value=20, value=3),
        'age_of_respondent': st.number_input('Age of Respondent', min_value=15, max_value=100, value=30),
        'gender_of_respondent': st.selectbox('Gender', choices['gender_of_respondent']),
        'relationship_with_head': st.selectbox('Relationship with Head', choices['relationship_with_head']),
        'marital_status': st.selectbox('Marital Status', choices['marital_status']),
        'education_level': st.selectbox('Education Level', choices['education_level']),
        'job_type': st.selectbox('Job Type', choices['job_type']),
    }
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Run the full pipeline (handles encoding + prediction)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Output result
        st.subheader("🔍 Prediction Result")
        if prediction == 1:
            st.success(f"✅ Likely to have a bank account (Probability: {probability:.1%})")
        else:
            st.error(f"❌ Unlikely to have a bank account (Probability: {probability:.1%})")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
