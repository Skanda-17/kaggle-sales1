import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
from models.preprocess import preprocess_data, load_and_clean_data

def train_model(data_path):
    """
    Train an XGBoost model for sales prediction
    
    Args:
        data_path: Path to the training data
        
    Returns:
        model: Trained XGBoost model
        scaler: Fitted scaler
        encoder: Fitted encoder
    """
    # Load and preprocess data
    df = load_and_clean_data(data_path)
    
    # Split features and target
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess data
    X_train_processed, scaler, encoder = preprocess_data(X_train, training=True)
    X_val_processed, _, _ = preprocess_data(X_val, scaler, encoder, training=False)
    
    # XGBoost parameters for better performance
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': 42
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train_processed, 
        y_train,
        eval_set=[(X_val_processed, y_val)],
        early_stopping_rounds=20,
        verbose=True
    )
    
    # Evaluate model
    val_preds = model.predict(X_val_processed)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mape = mean_absolute_percentage_error(y_val, val_preds) * 100
    
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation MAPE: {mape:.2f}%")
    
    return model, scaler, encoder

def predict_sales(model, X_processed):
    """
    Make sales predictions using trained model
    
    Args:
        model: Trained model
        X_processed: Preprocessed features
        
    Returns:
        Predicted sales value
    """
    prediction = model.predict(X_processed)
    # Ensure non-negative predictions
    prediction = np.maximum(0, prediction)
    return prediction
