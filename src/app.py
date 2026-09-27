
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border: none;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .churn-yes {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .churn-no {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

#TODO: Load Model Preprocessor
@st.cache_resource
def load_models():
    """Load the trained model and preprocessor"""

    try:
        # Project root
        project_root = Path(__file__).resolve().parent.parent

        # Models folder
        models_dir = project_root / "models"

        # Find trained model
        model_files = list(models_dir.glob("best_model_*.pkl"))

        if not model_files:
            st.error("❌ No trained model found in models folder!")
            return None, None

        # Select model
        model_path = model_files[0]

        # Preprocessor path
        preprocessor_path = models_dir / "preprocessor.pkl"

        # Check preprocessor
        if not preprocessor_path.exists():
            st.error(
                f"❌ Preprocessor not found:\n{preprocessor_path}"
            )
            return None, None

        st.write(f"Loading model: {model_path.name}")
        st.write(f"Loading preprocessor: {preprocessor_path.name}")

        # Load model
        model = joblib.load(model_path)

        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)

        st.success("✅ Model and preprocessor loaded successfully!")

        return model, preprocessor

    except Exception as e:

        st.error(f"❌ Error loading models: {str(e)}")

        import traceback
        st.code(traceback.format_exc())

        return None, None
    
def preprocess_input(input_df, preprocessor):
    """
    Preprocess input data using the same pipeline as training
    """
    # Make a copy to avoid modifying original
    df = input_df.copy()
    
    # Step 1: Handle missing values (if any)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isnull().sum() > 0:
        df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = df['TotalCharges'].median()
    
    # Step 2: Feature Engineering
    # Tenure groups
    df['tenure_group'] = pd.cut(
        df['tenure'], 
        bins=[-1, 12, 24, 48, 73],
        labels=[0, 1, 2, 3],
        include_lowest=True
    )
    df['tenure_group'] = df['tenure_group'].astype(float).astype(int)
    
    # Average monthly per tenure
    df['avg_monthly_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['avg_monthly_per_tenure'] = df['avg_monthly_per_tenure'].replace([np.inf, -np.inf], 0)
    
    # Number of services
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['num_services'] = 0
    for col in service_cols:
        if col in df.columns:
            df['num_services'] += (df[col] == 'Yes').astype(int)
    
    # Step 3: Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Step 4: Encode features
    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Gender encoding
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # Label encoding for remaining categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            # Handle unseen labels
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
        else:
            # If encoder doesn't exist, use a simple mapping
            df[col] = pd.Categorical(df[col]).codes
    
    # Step 5: Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_per_tenure']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    df[numerical_cols] = preprocessor.scaler.transform(df[numerical_cols])
    
    return df




#TODO:  Main App

def main():
    st.title("🔮 Customer Churn Prediction System")
    st.markdown("### Predict if a customer will churn based on their profile and usage patterns")
    st.markdown("---")
    
    #load modle
    model, preprocessor = load_models()
    if model is None or preprocessor is None:
        st.stop()
        
    #TODO: Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.markdown("""
        **Best Model:** Logistic Regression  
        **ROC-AUC Score:** 0.8458  
        **Accuracy:** 80.41%  
        **F1-Score:** 0.5929
        """)
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Fill in customer details on the right
        2. Click **Predict Churn** button
        3. View prediction and probability
        4. See feature importance
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Test Profiles")
        
        if st.button("Load High-Risk Profile"):
            st.session_state.test_profile = "high_risk"
        if st.button("Load Low-Risk Profile"):
            st.session_state.test_profile = "low_risk"

# Initialize session state for test profiles
    if 'test_profile' not in st.session_state:
        st.session_state.test_profile = None
        
        
     # Set default values based on test profile
    if st.session_state.test_profile == "high_risk":
        defaults = {
            'gender': 'Female', 'senior_citizen': 'No', 'partner': 'No', 'dependents': 'No',
            'tenure': 3, 'phone_service': 'Yes', 'multiple_lines': 'No',
            'internet_service': 'Fiber optic', 'online_security': 'No', 'online_backup': 'No',
            'device_protection': 'No', 'tech_support': 'No', 'streaming_tv': 'Yes',
            'streaming_movies': 'Yes', 'contract': 'Month-to-month', 'paperless_billing': 'Yes',
            'payment_method': 'Electronic check', 'monthly_charges': 85.0, 'total_charges': 255.0
        }
    elif st.session_state.test_profile == "low_risk":
        defaults = {
            'gender': 'Male', 'senior_citizen': 'No', 'partner': 'Yes', 'dependents': 'Yes',
            'tenure': 48, 'phone_service': 'Yes', 'multiple_lines': 'Yes',
            'internet_service': 'Fiber optic', 'online_security': 'Yes', 'online_backup': 'Yes',
            'device_protection': 'Yes', 'tech_support': 'Yes', 'streaming_tv': 'Yes',
            'streaming_movies': 'Yes', 'contract': 'Two year', 'paperless_billing': 'No',
            'payment_method': 'Credit card (automatic)', 'monthly_charges': 105.0, 'total_charges': 5040.0
        }
    else:
        defaults = {
            'gender': 'Male', 'senior_citizen': 'No', 'partner': 'No', 'dependents': 'No',
            'tenure': 12, 'phone_service': 'Yes', 'multiple_lines': 'No',
            'internet_service': 'DSL', 'online_security': 'No', 'online_backup': 'No',
            'device_protection': 'No', 'tech_support': 'No', 'streaming_tv': 'No',
            'streaming_movies': 'No', 'contract': 'Month-to-month', 'paperless_billing': 'Yes',
            'payment_method': 'Electronic check', 'monthly_charges': 70.0, 'total_charges': 840.0
        }
    
        
    col1, col2 = st.columns([2,1])  
    
    with col1:
        st.header("📝 Customer Information")
        # Create input form
        with st.form("customer_form"):
            # Demographics
            st.subheader("👤 Demographics")
            demo_col1, demo_col2, demo_col3 = st.columns(3)
            
            with demo_col1:
                gender = st.selectbox("Gender", ["Male", "Female"], 
                                     index=0 if defaults['gender']=="Male" else 1)
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"],
                                             index=0 if defaults['senior_citizen']=="No" else 1)
            
            with demo_col2:
                partner = st.selectbox("Has Partner", ["No", "Yes"],
                                      index=0 if defaults['partner']=="No" else 1)
                dependents = st.selectbox("Has Dependents", ["No", "Yes"],
                                         index=0 if defaults['dependents']=="No" else 1)
            
            with demo_col3:
                tenure = st.slider("Tenure (months)", 0, 72, defaults['tenure'])
            
            st.markdown("---")
            
            # Account Information
            st.subheader("💳 Account Information")
            acc_col1, acc_col2, acc_col3 = st.columns(3)
            
            contract_options = ["Month-to-month", "One year", "Two year"]
            with acc_col1:
                contract = st.selectbox("Contract Type", contract_options,
                                       index=contract_options.index(defaults['contract']))
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"],
                                                index=0 if defaults['paperless_billing']=="No" else 1)
            
            payment_options = ["Electronic check", "Mailed check", 
                              "Bank transfer (automatic)", "Credit card (automatic)"]
            with acc_col2:
                payment_method = st.selectbox("Payment Method", payment_options,
                                             index=payment_options.index(defaults['payment_method']))
            
            with acc_col3:
                monthly_charges = st.number_input("Monthly Charges ($)", 
                                                 min_value=0.0, 
                                                 max_value=200.0, 
                                                 value=defaults['monthly_charges'], 
                                                 step=5.0)
                total_charges = st.number_input("Total Charges ($)", 
                                               min_value=0.0, 
                                               max_value=10000.0, 
                                               value=defaults['total_charges'], 
                                               step=50.0)
        
        
            st.markdown("---")
            
            # Services
            st.subheader("📞 Services")
            serv_col1, serv_col2 = st.columns(2)
            
            with serv_col1:
                phone_service = st.selectbox("Phone Service", ["No", "Yes"],
                                            index=0 if defaults['phone_service']=="No" else 1)
                multiple_lines = st.selectbox("Multiple Lines", 
                                             ["No", "Yes", "No phone service"],
                                             index=["No", "Yes", "No phone service"].index(defaults['multiple_lines']))
                
                internet_options = ["DSL", "Fiber optic", "No"]
                internet_service = st.selectbox("Internet Service", internet_options,
                                               index=internet_options.index(defaults['internet_service']))
            
            with serv_col2:
                online_security = st.selectbox("Online Security", 
                                              ["No", "Yes", "No internet service"],
                                              index=["No", "Yes", "No internet service"].index(defaults['online_security']))
                online_backup = st.selectbox("Online Backup", 
                                            ["No", "Yes", "No internet service"],
                                            index=["No", "Yes", "No internet service"].index(defaults['online_backup']))
                device_protection = st.selectbox("Device Protection", 
                                                ["No", "Yes", "No internet service"],
                                                index=["No", "Yes", "No internet service"].index(defaults['device_protection']))
            
            serv_col3, serv_col4 = st.columns(2)
            
            with serv_col3:
                tech_support = st.selectbox("Tech Support", 
                                           ["No", "Yes", "No internet service"],
                                           index=["No", "Yes", "No internet service"].index(defaults['tech_support']))
            
            with serv_col4:
                streaming_tv = st.selectbox("Streaming TV", 
                                           ["No", "Yes", "No internet service"],
                                           index=["No", "Yes", "No internet service"].index(defaults['streaming_tv']))
                streaming_movies = st.selectbox("Streaming Movies", 
                                               ["No", "Yes", "No internet service"],
                                               index=["No", "Yes", "No internet service"].index(defaults['streaming_movies']))
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)       
        
        with col2:
            st.header("🎯 Prediction Results")
            
            if submitted:
                # Reset test profile after submission
                st.session_state.test_profile = None
                
                # Create input dataframe
                input_data = pd.DataFrame({
                    'customerID': ['PRED-001'],
                    'gender': [gender],
                    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges]
                })
                
                try:
                    # Preprocess the input
                    processed_data = preprocess_input(input_data, preprocessor)
                    
                    # Debug: Show processed data shape
                    with st.expander("🔍 Debug Info (Click to expand)"):
                        st.write("**Processed data shape:**", processed_data.shape)
                        st.write("**Processed columns:**", processed_data.columns.tolist())
                        st.write("**Sample values:**")
                        st.dataframe(processed_data.head())
                    
                    # Make prediction
                    prediction = model.predict(processed_data)[0]
                    prediction_proba = model.predict_proba(processed_data)[0]
                    
                    # Display results
                    churn_probability = prediction_proba[1] * 100
                    no_churn_probability = prediction_proba[0] * 100
                    
                    # Prediction box
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-box churn-yes">
                            <h2 style="color: #f44336; margin: 0;">⚠️ HIGH RISK</h2>
                            <h3 style="margin: 0.5rem 0;">Customer Likely to Churn</h3>
                            <h1 style="color: #f44336; margin: 0;">{churn_probability:.1f}%</h1>
                            <p style="margin: 0.5rem 0;">Churn Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("**Recommendation:** Implement retention strategies immediately!")
                        st.markdown("""
                        **Suggested Actions:**
                        - Offer special discount or promotion
                        - Reach out to customer support team
                        - Propose contract upgrade with benefits
                        - Survey customer satisfaction
                        """)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box churn-no">
                            <h2 style="color: #4caf50; margin: 0;">✅ LOW RISK</h2>
                            <h3 style="margin: 0.5rem 0;">Customer Likely to Stay</h3>
                            <h1 style="color: #4caf50; margin: 0;">{no_churn_probability:.1f}%</h1>
                            <p style="margin: 0.5rem 0;">Retention Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("**Status:** Customer retention looks good!")
                        st.markdown("""
                        **Continue to:**
                        - Maintain service quality
                        - Engage with loyalty programs
                        - Regular satisfaction checks
                        """)
                    
                    # Probability gauge chart
                    st.markdown("### 📊 Confidence Breakdown")
                    
                    fig = go.Figure(go.Bar(
                        x=[no_churn_probability, churn_probability],
                        y=['Will Stay', 'Will Churn'],
                        orientation='h',
                        marker=dict(color=['#4caf50', '#f44336']),
                        text=[f'{no_churn_probability:.1f}%', f'{churn_probability:.1f}%'],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(range=[0, 100], showticklabels=False),
                        yaxis=dict(showticklabels=True),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk factors
                    st.markdown("### 🎯 Key Risk Factors")
                    
                    risk_factors = []
                    if contract == "Month-to-month":
                        risk_factors.append("⚠️ Month-to-month contract")
                    if tenure < 12:
                        risk_factors.append("⚠️ Short tenure (< 1 year)")
                    if monthly_charges > 80:
                        risk_factors.append("⚠️ High monthly charges")
                    if online_security == "No":
                        risk_factors.append("⚠️ No online security")
                    if tech_support == "No":
                        risk_factors.append("⚠️ No tech support")
                    if payment_method == "Electronic check":
                        risk_factors.append("⚠️ Electronic check payment")
                    if partner == "No":
                        risk_factors.append("⚠️ No partner")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(f"- {factor}")
                    else:
                        st.markdown("✅ No major risk factors identified")
                    
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.info("👈 Fill in the customer details and click **Predict Churn** to see results")
                
                # Show example
                st.markdown("### 💡 Quick Start")
                st.markdown("""
                Use the sidebar buttons to load test profiles:
                - **High-Risk Profile**: New customer, month-to-month, no services
                - **Low-Risk Profile**: Long tenure, 2-year contract, all services
                """)


if __name__ == "__main__":
    main()