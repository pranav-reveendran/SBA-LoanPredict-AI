from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables to store loaded models
model = None
scaler = None
features = None
model_results = None
model_summary = None

def load_models():
    """Load all trained models and artifacts"""
    global model, scaler, features, model_results, model_summary
    
    try:
        model = joblib.load('trained_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        model_results = joblib.load('model_results.pkl')
        
        # Load model summary if exists
        if os.path.exists('model_summary.json'):
            with open('model_summary.json', 'r') as f:
                model_summary = json.load(f)
        
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "SBA Loan Default Prediction API",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model is not None
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about available models and their performance"""
    if model_results is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    # Convert model results to JSON-serializable format
    models_info = {}
    for name, result in model_results.items():
        models_info[name] = {
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1_score']),
            'auc': float(result['auc'])
        }
    
    response = {
        'models': models_info,
        'best_model': model_summary['best_model'] if model_summary else 'Unknown',
        'features': features,
        'total_models': len(models_info)
    }
    
    if model_summary and 'profit_analysis' in model_summary:
        response['profit_analysis'] = model_summary['profit_analysis']
    
    return jsonify(response)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction for a single loan application"""
    if model is None or scaler is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract features in the correct order
        feature_values = []
        for feature in features:
            if feature in data:
                feature_values.append(float(data[feature]))
            else:
                # Use default values for missing features
                default_values = {
                    'Term': 120,
                    'NoEmp': 10,
                    'CreateJob': 0,
                    'RetainedJob': 0,
                    'DisbursementGross': 250000,
                    'GrAppv': 275000,
                    'SBA_Appv': 200000,
                    'NewExist': 1,
                    'UrbanRural': 1,
                    'RevLineCr': 0,
                    'LowDoc': 1,
                    'NAICS': 561
                }
                feature_values.append(float(default_values.get(feature, 0)))
        
        # Create feature array
        X = np.array([feature_values])
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_scaled)[0]
        default_probability = float(prediction_proba[1])
        
        # Use optimal threshold from model training
        threshold = model_summary.get('optimal_threshold', 0.4) if model_summary else 0.4
        prediction = int(default_probability > threshold)
        
        # Calculate risk level
        if default_probability > 0.7:
            risk_level = "High"
        elif default_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate expected value (simplified business logic)
        loan_amount = data.get('DisbursementGross', 250000)
        profit_per_good_loan = loan_amount * 0.04  # 4% profit margin
        loss_per_bad_loan = loan_amount * 0.8  # 80% loss if default
        
        expected_value = profit_per_good_loan * (1 - default_probability) - loss_per_bad_loan * default_probability
        
        response = {
            "prediction": {
                "default_probability": default_probability,
                "recommend_approval": prediction == 0,  # Approve if not predicted to default
                "risk_level": risk_level,
                "expected_value": float(expected_value),
                "threshold_used": threshold
            },
            "input_features": dict(zip(features, feature_values)),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Make predictions for multiple loan applications"""
    if model is None or scaler is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'loans' not in data:
            return jsonify({"error": "No loan data provided"}), 400
        
        loans = data['loans']
        predictions = []
        
        for i, loan_data in enumerate(loans):
            # Extract features
            feature_values = []
            for feature in features:
                if feature in loan_data:
                    feature_values.append(float(loan_data[feature]))
                else:
                    # Use default values
                    default_values = {
                        'Term': 120, 'NoEmp': 10, 'CreateJob': 0, 'RetainedJob': 0,
                        'DisbursementGross': 250000, 'GrAppv': 275000, 'SBA_Appv': 200000,
                        'NewExist': 1, 'UrbanRural': 1, 'RevLineCr': 0, 'LowDoc': 1, 'NAICS': 561
                    }
                    feature_values.append(float(default_values.get(feature, 0)))
            
            # Make prediction
            X = np.array([feature_values])
            X_scaled = scaler.transform(X)
            
            prediction_proba = model.predict_proba(X_scaled)[0]
            default_probability = float(prediction_proba[1])
            
            threshold = 0.4
            prediction = int(default_probability > threshold)
            
            # Risk level
            if default_probability > 0.7:
                risk_level = "High"
            elif default_probability > 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Expected value
            loan_amount = loan_data.get('DisbursementGross', 250000)
            profit_per_good_loan = loan_amount * 0.04
            loss_per_bad_loan = loan_amount * 0.8
            expected_value = profit_per_good_loan * (1 - default_probability) - loss_per_bad_loan * default_probability
            
            predictions.append({
                "loan_id": i,
                "default_probability": default_probability,
                "recommend_approval": prediction == 0,
                "risk_level": risk_level,
                "expected_value": float(expected_value)
            })
        
        # Calculate summary statistics
        total_loans = len(predictions)
        approved_loans = sum(1 for p in predictions if p['recommend_approval'])
        approval_rate = approved_loans / total_loans if total_loans > 0 else 0
        avg_default_prob = sum(p['default_probability'] for p in predictions) / total_loans if total_loans > 0 else 0
        total_expected_value = sum(p['expected_value'] for p in predictions)
        
        response = {
            "predictions": predictions,
            "summary": {
                "total_loans": total_loans,
                "approved_loans": approved_loans,
                "approval_rate": approval_rate,
                "average_default_probability": avg_default_prob,
                "total_expected_value": total_expected_value
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get model analytics and performance metrics"""
    if model_summary is None:
        return jsonify({"error": "Model summary not available"}), 500
    
    return jsonify({
        "model_summary": model_summary,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of required features for prediction"""
    if features is None:
        return jsonify({"error": "Features not loaded"}), 500
    
    feature_descriptions = {
        'Term': 'Loan term in months',
        'NoEmp': 'Number of employees',
        'CreateJob': 'Number of jobs created',
        'RetainedJob': 'Number of jobs retained',
        'DisbursementGross': 'Gross loan amount disbursed ($)',
        'GrAppv': 'Gross amount approved ($)',
        'SBA_Appv': 'SBA guaranteed amount ($)',
        'NewExist': 'Business type (1=Existing, 2=New)',
        'UrbanRural': 'Location type (0=Undefined, 1=Urban, 2=Rural)',
        'RevLineCr': 'Revolving line of credit (0=No, 1=Yes)',
        'LowDoc': 'Low documentation program (0=No, 1=Yes)',
        'NAICS': 'NAICS industry code'
    }
    
    return jsonify({
        "features": features,
        "feature_descriptions": feature_descriptions,
        "total_features": len(features)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting SBA Loan Default Prediction API...")
    
    # Load models on startup
    if load_models():
        print("All models loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to load models. Please run train_model.py first.")
        print("Starting API anyway for testing purposes...")
        app.run(debug=True, host='0.0.0.0', port=5001) 