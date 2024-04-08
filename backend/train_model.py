import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def clean_currency_column(series):
    """Clean currency columns by removing $ and commas"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace(' ', '', regex=False)
        .replace('', '0'), 
        errors='coerce'
    ).fillna(0)

def preprocess_data(df):
    """Preprocess the real SBA loan data"""
    print("Preprocessing SBA data...")
    print(f"Original dataset shape: {df.shape}")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Clean currency columns
    currency_cols = ['DisbursementGross', 'GrAppv', 'SBA_Appv', 'BalanceGross', 'ChgOffPrinGr']
    for col in currency_cols:
        if col in df.columns:
            df[col] = clean_currency_column(df[col])
    
    # Create binary target variable from MIS_Status
    df['Default'] = (df['MIS_Status'] == 'CHGOFF').astype(int)
    
    # Handle categorical variables
    # RevLineCr: Convert to binary (Y=1, N=0, others=0)
    df['RevLineCr_Binary'] = (df['RevLineCr'] == 'Y').astype(int)
    
    # LowDoc: Convert to binary (Y=1, N=0, others=0)  
    df['LowDoc_Binary'] = (df['LowDoc'] == 'Y').astype(int)
    
    # NewExist: Handle missing values and ensure proper encoding
    df['NewExist'] = df['NewExist'].fillna(1)  # Default to existing business
    df['NewExist'] = df['NewExist'].astype(int)
    
    # UrbanRural: Already numeric, handle any missing
    df['UrbanRural'] = df['UrbanRural'].fillna(0)  # Default to undefined
    
    # NAICS: Handle missing values
    df['NAICS'] = df['NAICS'].fillna(0)  # 0 for missing industry codes
    
    # Define features for modeling (based on competition requirements)
    features = [
        'Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'DisbursementGross',
        'GrAppv', 'SBA_Appv', 'NewExist', 'UrbanRural', 'RevLineCr_Binary', 
        'LowDoc_Binary', 'NAICS'
    ]
    
    # Handle missing values in numerical features
    numerical_features = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'DisbursementGross', 'GrAppv', 'SBA_Appv']
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Remove rows with missing target variable
    df = df.dropna(subset=['Default'])
    
    # Create feature matrix
    X = df[features].copy()
    y = df['Default'].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    print(f"Processed dataset shape: {X.shape}")
    print(f"Default rate: {y.mean():.3%}")
    print(f"Features: {features}")
    
    return X, y, features

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return the best one"""
    print("Training multiple models...")
    
    # Define models
    models = {
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        print(f"{name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    
    # Find best model (based on F1 score for imbalanced data)
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    
    return best_model, results, best_model_name

def calculate_profit_analysis(model, X_test, y_test, disbursement_amounts, threshold=0.4):
    """Calculate profit analysis using the competition's cost structure"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Competition cost structure:
    # Profit per good loan = 5% of DisbursementGross
    # Loss per bad loan = -5 * 5% of DisbursementGross = -25% of DisbursementGross
    
    total_profit = 0
    approved_loans = 0
    
    for i in range(len(y_test)):
        loan_amount = disbursement_amounts.iloc[i]
        actual_default = y_test.iloc[i]
        predicted_approve = (y_pred[i] == 0)  # Approve if predicted not to default
        
        if predicted_approve:  # We approve the loan
            approved_loans += 1
            if actual_default == 0:  # Loan paid in full
                total_profit += 0.05 * loan_amount  # 5% profit
            else:  # Loan defaulted
                total_profit -= 0.25 * loan_amount  # 25% loss (5 * 5%)
        # If we deny the loan, profit = 0
    
    return {
        'expected_profit': total_profit,
        'threshold': threshold,
        'approval_rate': approved_loans / len(y_test),
        'total_loans_evaluated': len(y_test),
        'loans_approved': approved_loans
    }

def main():
    """Main training pipeline"""
    print("Starting SBA Loan Default Prediction Model Training...")
    
    # Load the real SBA data
    data_path = '../data/sba_national_data.csv'
    print(f"Loading SBA National Dataset from {data_path}")
    
    # Load with low_memory=False to handle mixed types
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded dataset with {len(df):,} observations and {len(df.columns)} variables")
    
    # Preprocess data
    X, y, features = preprocess_data(df)
    
    # Split data (using stratified split to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("Training multiple models...")
    best_model, results, best_model_name = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Get disbursement amounts for profit analysis
    disbursement_test = X_test['DisbursementGross'].reset_index(drop=True)
    
    # Profit analysis with different thresholds
    print("\nTesting different thresholds for profit optimization...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_profit = float('-inf')
    best_threshold = 0.4
    
    for threshold in thresholds:
        profit_analysis = calculate_profit_analysis(best_model, X_test_scaled, y_test, disbursement_test, threshold)
        print(f"Threshold {threshold}: Profit=${profit_analysis['expected_profit']:,.2f}, Approval Rate={profit_analysis['approval_rate']:.1%}")
        
        if profit_analysis['expected_profit'] > best_profit:
            best_profit = profit_analysis['expected_profit']
            best_threshold = threshold
    
    # Final profit analysis with best threshold
    final_profit_analysis = calculate_profit_analysis(best_model, X_test_scaled, y_test, disbursement_test, best_threshold)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best Model: {best_model_name}")
    print(f"Optimal Threshold: {best_threshold}")
    print(f"Expected Profit: ${final_profit_analysis['expected_profit']:,.2f}")
    print(f"Approval Rate: {final_profit_analysis['approval_rate']:.1%}")
    print(f"Loans Approved: {final_profit_analysis['loans_approved']:,} out of {final_profit_analysis['total_loans_evaluated']:,}")
    
    # Save model and artifacts
    print("\nSaving models and artifacts...")
    joblib.dump(best_model, 'trained_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'features.pkl')
    joblib.dump(results, 'model_results.pkl')
    
    # Save model summary
    summary = {
        'best_model': best_model_name,
        'optimal_threshold': best_threshold,
        'features': features,
        'model_performance': {k: {metric: v for metric, v in results[k].items() if metric != 'model'} 
                             for k in results.keys()},
        'profit_analysis': final_profit_analysis,
        'dataset_info': {
            'total_samples': len(df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'default_rate': float(y.mean()),
            'features_used': len(features)
        }
    }
    
    import json
    with open('model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Model saved as 'trained_model.pkl'")
    print(f"Scaler saved as 'scaler.pkl'")
    print(f"Features saved as 'features.pkl'")
    print(f"Results saved as 'model_results.pkl'")
    print(f"Summary saved as 'model_summary.json'")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 