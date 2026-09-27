import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders={}
        
        
    def load_data(self, filepath):
        print(f"Laoding data from {filepath} ...")
        df = pd.read_csv(filepath)
        print("\n Data Loaded")
        return df
    
    def handling_missing_values(self, df):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        if df['TotalCharges'].isnull().sum()>0:
            df.fillna({'TotalCharges':df['TotalCharges'].median() }, inplace=True)
            
            print(f"Filled missing values ")
            
        return df
    
    def feature_engineering(self,df):
        """Create new features"""
        print("\nCreating new features...")
        
        # Check tenure range
        print(f"Tenure range: {df['tenure'].min()} to {df['tenure'].max()}")
        
        # Tenure groups - FIX: Handle edge cases and NaN
        # Use right=True to include right edge, and extend bins to cover all values
        df['tenure_group'] = pd.cut(
            df['tenure'], 
            bins=[-1, 12, 24, 48, 73],  # Extended range to cover 0-72
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        
        # Convert to float first (handles NaN), then to int
        df['tenure_group'] = df['tenure_group'].astype(float).astype(int)
        
        # Average monthly charge per tenure (avoid division by zero)
        df['avg_monthly_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Replace any inf values with 0
        df['avg_monthly_per_tenure'] = df['avg_monthly_per_tenure'].replace([np.inf, -np.inf], 0)
        
        # Has multiple services
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport']
        
        # Count services (handling 'No internet service' properly)
        df['num_services'] = 0
        for col in service_cols:
            if col in df.columns:
                # Count as service if value is 'Yes'
                df['num_services'] += (df[col] == 'Yes').astype(int)
        
        print(f"✅ Created 3 new features")
        print(f"   - tenure_group: {df['tenure_group'].unique()}")
        print(f"   - avg_monthly_per_tenure: min={df['avg_monthly_per_tenure'].min():.2f}, max={df['avg_monthly_per_tenure'].max():.2f}")
        print(f"   - num_services: min={df['num_services'].min()}, max={df['num_services'].max()}")
        
        return df
           
    def encode_features(self, df, is_training=False):

        # 1. Remove customerID
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # 2. Binary columns: Yes/No -> 1/0
        binary_cols = [
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling'
    ]

        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].map({
                'Yes': 1,
                'No': 0
            })

        # 3. Gender: Male/Female -> 1/0
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.strip()
            df['gender'] = df['gender'].map({
            'Male': 1,
            'Female': 0
        })

    # 4. Target variable: Yes/No -> 1/0
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].astype(str).str.strip()
            df['Churn'] = df['Churn'].map({
            'Yes': 1,
            'No': 0
        })

    # 5. Find remaining categorical columns
        categorical_cols = df.select_dtypes(
        include=['object']
    ).columns.tolist()

    # 6. Encode remaining categorical columns
        for col in categorical_cols:

            if is_training:
                le = LabelEncoder()

                df[col] = le.fit_transform(
                df[col].astype(str).str.strip()
            )

                self.label_encoders[col] = le

                print(
                f"   - Encoded {col}: "
                f"{len(le.classes_)} classes"
            )

            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]

                    df[col] = le.transform(
                    df[col].astype(str).str.strip()
                )

    # 7. Final check
        object_cols = df.select_dtypes(
            include=['object']
    ).columns.tolist()

        print("\nRemaining object columns:")
        print(object_cols)

        print("\nEncoded categorical features successfully")

        return df            
    def scale_features(self, X_train, X_test = None, is_training=True):
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_per_tenure']
        
        numerical_cols = [col for col in numerical_cols if col in X_train.columns]
        
        
        if is_training:
            X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            if X_test is not None:
                X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
                
        else:
            X_train[numerical_cols] = self.scaler.transform(X_train[numerical_cols])
            
        print("FEATURES SCALED!!!")
        return X_train,X_test
       
    #TODO: Preprocessing Pipeline
    def prepare_data(self, df, test_size=0.2, random_state=42):
        
        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        #handling missing values
        df = self.handling_missing_values(df)
        
        #feature engineering
        df = self.feature_engineering(df)
        
        #encoding
        df = self.encode_features(df, True)
        
        #split Features and Target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        #traintest
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)
        
        print(f"\n📊 Train-Test Split:")
        print(f"   Train set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"   Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
        
        #scale Features
        
        X_train, X_test = self.scale_features(X_train, X_test, is_training=True)
    
    
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE")
        print("="*60)
        
        return X_train, X_test, y_train, y_test
    
    

    
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    df = preprocessor.load_data("C:/Users/COMPUTER AGE#/Desktop/Churn-prediction-ML-project/Data/telco_comm_churn.csv")
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)    
    
    
    print(f"\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"✅ Training features: {X_train.shape}")
    print(f"✅ Test features: {X_test.shape}")
    print(f"✅ Training target: {y_train.shape}")
    print(f"✅ Test target: {y_test.shape}")
    
    # Verify no object columns remain
    print(f"\n📊 Final Data Types:")
    print(X_train.dtypes.value_counts())
    
    # Sample of first few rows
    print(f"\n📋 Sample Data (first 3 rows, first 5 columns):")
    print(X_train.iloc[:3, :5])
        