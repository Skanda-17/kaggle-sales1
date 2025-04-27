import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_clean_data(file_path):
    """
    Load and clean the Rossmann store sales data
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        df: Cleaned DataFrame
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Basic cleaning
    df = df.fillna(0)
    
    # Filter out closed stores (sales would be 0)
    df = df[df['Open'] != 0]
    
    # Make sure StateHoliday is a string
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    
    # Extract date components
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df = df.drop('Date', axis=1)
    
    # Drop any unnecessary columns
    cols_to_drop = ['Customers']  # We usually don't have this info when predicting
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df

def preprocess_data(df, scaler=None, encoder=None, training=True):
    """
    Preprocess data for model training or prediction
    
    Args:
        df: Input DataFrame
        scaler: StandardScaler object (optional, used for test data)
        encoder: OneHotEncoder object (optional, used for test data)
        training: Whether this is for training or prediction
        
    Returns:
        X_processed: Processed features ready for model
        scaler: Fitted scaler (only if training=True)
        encoder: Fitted encoder (only if training=True)
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    data = data.fillna(0)
    
    # Separate numerical and categorical features
    num_features = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # For numeric features: scaling
    if training:
        scaler = StandardScaler()
        data[num_features] = scaler.fit_transform(data[num_features])
    else:
        data[num_features] = scaler.transform(data[num_features])
    
    # For categorical features: one-hot encoding
    if len(cat_features) > 0:
        if training:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            cat_encoded = encoder.fit_transform(data[cat_features])
        else:
            cat_encoded = encoder.transform(data[cat_features])
        
        # Convert encoded features to DataFrame
        cat_encoded_df = pd.DataFrame(
            cat_encoded,
            columns=encoder.get_feature_names_out(cat_features),
            index=data.index
        )
        
        # Drop original categorical columns and join with encoded ones
        data = data.drop(cat_features, axis=1)
        data = pd.concat([data, cat_encoded_df], axis=1)
    
    if training:
        return data, scaler, encoder
    else:
        return data
