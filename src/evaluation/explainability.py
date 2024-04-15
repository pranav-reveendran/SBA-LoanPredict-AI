"""
SHAP-based Model Explainability for SBA Loan Prediction
Provides comprehensive model interpretation for regulatory compliance
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple
import warnings
import pickle
import joblib

warnings.filterwarnings('ignore')

class SHAPExplainer:
    """
    Comprehensive SHAP-based model explanation system
    """
    
    def __init__(self, model, feature_names: List[str], model_type: str = 'tree'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model object
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.base_value = None
    
    def initialize_explainer(self, X_background: np.ndarray, max_evals: int = 100):
        """
        Initialize the appropriate SHAP explainer
        
        Args:
            X_background: Background dataset for explanation
            max_evals: Maximum evaluations for kernel explainer
        """
        if self.model_type == 'tree':
            # For tree-based models (Random Forest, XGBoost, etc.)
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            # For linear models
            self.explainer = shap.LinearExplainer(self.model, X_background)
        else:
            # For any other model type
            def model_predict(X):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)[:, 1]
                else:
                    return self.model.predict(X)
            
            self.explainer = shap.KernelExplainer(model_predict, X_background, 
                                                link="logit")
    
    def calculate_shap_values(self, X_explain: np.ndarray):
        """
        Calculate SHAP values for explanation dataset
        
        Args:
            X_explain: Dataset to explain
        """
        if self.explainer is None:
            raise ValueError("Must initialize explainer first")
        
        self.shap_values = self.explainer.shap_values(X_explain)
        
        # Handle different output formats
        if isinstance(self.shap_values, list):
            # For classification models that return list
            self.shap_values = self.shap_values[1]  # Use positive class
        
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                self.base_value = self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0]
            else:
                self.base_value = self.explainer.expected_value
        else:
            self.base_value = 0
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive feature importance plot
        
        Args:
            top_n: Number of top features to plot
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        importance_df = self.get_feature_importance(top_n)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(color='lightblue'),
            hovertemplate='<b>%{y}</b><br>' +
                         'Importance: %{x:.4f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance (Mean |SHAP|)',
            xaxis_title='Mean Absolute SHAP Value',
            yaxis_title='Features',
            height=max(400, top_n * 25),
            width=800,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def explain_prediction(self, instance_idx: int, X_explain: np.ndarray) -> Dict:
        """
        Explain individual prediction
        
        Args:
            instance_idx: Index of instance to explain
            X_explain: Dataset containing the instance
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        instance_shap = self.shap_values[instance_idx]
        instance_features = X_explain[instance_idx]
        
        # Create feature contribution dataframe
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'feature_value': instance_features,
            'shap_value': instance_shap,
            'abs_shap': np.abs(instance_shap)
        }).sort_values('abs_shap', ascending=False)
        
        # Calculate prediction
        prediction_score = self.base_value + np.sum(instance_shap)
        
        return {
            'prediction_score': prediction_score,
            'base_value': self.base_value,
            'shap_sum': np.sum(instance_shap),
            'contributions': contributions,
            'top_positive': contributions[contributions['shap_value'] > 0].head(5),
            'top_negative': contributions[contributions['shap_value'] < 0].head(5)
        }
    
    def plot_waterfall(self, instance_idx: int, X_explain: np.ndarray, 
                      max_features: int = 10, save_path: Optional[str] = None) -> go.Figure:
        """
        Create waterfall plot for individual prediction explanation
        
        Args:
            instance_idx: Index of instance to explain
            X_explain: Dataset containing the instance
            max_features: Maximum number of features to show
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        explanation = self.explain_prediction(instance_idx, X_explain)
        contributions = explanation['contributions'].head(max_features)
        
        # Prepare data for waterfall chart
        features = ['Base'] + list(contributions['feature']) + ['Prediction']
        values = [explanation['base_value']] + list(contributions['shap_value']) + [explanation['prediction_score']]
        
        # Calculate cumulative values for positioning
        cumulative = [explanation['base_value']]
        for i, val in enumerate(contributions['shap_value']):
            cumulative.append(cumulative[-1] + val)
        cumulative.append(explanation['prediction_score'])
        
        # Create colors (green for positive, red for negative, blue for base/prediction)
        colors = ['blue']
        for val in contributions['shap_value']:
            colors.append('green' if val >= 0 else 'red')
        colors.append('blue')
        
        fig = go.Figure()
        
        # Add bars
        for i, (feature, value, color) in enumerate(zip(features, values, colors)):
            if i == 0 or i == len(features) - 1:
                # Base and prediction bars
                fig.add_trace(go.Bar(
                    x=[feature],
                    y=[value],
                    marker_color=color,
                    name=feature,
                    showlegend=False,
                    hovertemplate=f'<b>{feature}</b><br>Value: {value:.4f}<extra></extra>'
                ))
            else:
                # Contribution bars
                base_y = cumulative[i-1] if value >= 0 else cumulative[i]
                fig.add_trace(go.Bar(
                    x=[feature],
                    y=[abs(value)],
                    base=base_y,
                    marker_color=color,
                    name=feature,
                    showlegend=False,
                    hovertemplate=f'<b>{feature}</b><br>Contribution: {value:+.4f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'SHAP Waterfall Plot - Instance {instance_idx}',
            xaxis_title='Features',
            yaxis_title='Model Output',
            height=600,
            width=1000,
            xaxis={'tickangle': 45}
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_summary(self, X_explain: np.ndarray, max_features: int = 20, 
                    save_path: Optional[str] = None) -> go.Figure:
        """
        Create summary plot showing feature importance and value distributions
        
        Args:
            X_explain: Dataset for explanation
            max_features: Maximum number of features to show
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Get top features by importance
        importance_df = self.get_feature_importance(max_features)
        top_features = importance_df['feature'].tolist()
        
        # Get indices of top features
        feature_indices = [self.feature_names.index(f) for f in top_features]
        
        # Create subplot for each feature
        fig = make_subplots(
            rows=len(top_features), cols=1,
            subplot_titles=top_features,
            vertical_spacing=0.02
        )
        
        for i, (feature_name, feature_idx) in enumerate(zip(top_features, feature_indices)):
            # SHAP values for this feature
            feature_shap = self.shap_values[:, feature_idx]
            feature_values = X_explain[:, feature_idx]
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=feature_shap,
                    y=[feature_name] * len(feature_shap),
                    mode='markers',
                    marker=dict(
                        color=feature_values,
                        colorscale='RdYlBu_r',
                        size=3,
                        opacity=0.6
                    ),
                    name=feature_name,
                    showlegend=False,
                    hovertemplate=f'<b>{feature_name}</b><br>' +
                                 'SHAP Value: %{x:.4f}<br>' +
                                 'Feature Value: %{marker.color:.2f}<br>' +
                                 '<extra></extra>'
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title='SHAP Summary Plot - Feature Impact Distribution',
            xaxis_title='SHAP Value (Impact on Model Output)',
            height=max(400, len(top_features) * 40),
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

class RegulatoryReportGenerator:
    """
    Generate regulatory compliance reports for model explainability
    """
    
    def __init__(self, shap_explainer: SHAPExplainer):
        """
        Initialize with SHAP explainer
        
        Args:
            shap_explainer: Initialized SHAP explainer object
        """
        self.explainer = shap_explainer
    
    def generate_model_card(self, model_performance: Dict, 
                          feature_importance: pd.DataFrame) -> Dict:
        """
        Generate model card for regulatory documentation
        
        Args:
            model_performance: Dictionary with model performance metrics
            feature_importance: DataFrame with feature importance
            
        Returns:
            Dictionary with model card information
        """
        return {
            'model_overview': {
                'purpose': 'SBA Loan Default Risk Assessment',
                'model_type': 'Machine Learning Classifier',
                'regulatory_framework': 'Basel III Compliance',
                'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d')
            },
            'performance_metrics': model_performance,
            'feature_importance': {
                'top_10_features': feature_importance.head(10).to_dict('records'),
                'methodology': 'SHAP (SHapley Additive exPlanations)',
                'interpretation': 'Higher absolute values indicate greater impact on default prediction'
            },
            'risk_factors': {
                'primary_risk_indicators': feature_importance.head(5)['feature'].tolist(),
                'model_limitations': [
                    'Performance may degrade with economic regime changes',
                    'Feature distributions should be monitored for drift',
                    'Model requires periodic retraining'
                ]
            },
            'compliance_notes': {
                'explainability': 'Model provides individual prediction explanations via SHAP',
                'fairness': 'Regular bias testing recommended',
                'auditability': 'All predictions can be traced to feature contributions'
            }
        }
    
    def generate_decision_explanation(self, loan_application: Dict, 
                                    shap_explanation: Dict) -> Dict:
        """
        Generate human-readable explanation for loan decision
        
        Args:
            loan_application: Dictionary with loan application details
            shap_explanation: SHAP explanation from explain_prediction
            
        Returns:
            Dictionary with decision explanation
        """
        top_positive = shap_explanation['top_positive']
        top_negative = shap_explanation['top_negative']
        
        # Determine decision
        prediction_score = shap_explanation['prediction_score']
        decision = 'APPROVE' if prediction_score < 0.4 else 'DENY'  # Assuming 0.4 threshold
        
        explanation = {
            'decision': decision,
            'confidence_score': abs(prediction_score),
            'key_factors': {
                'supporting_approval': [],
                'supporting_denial': []
            },
            'detailed_explanation': '',
            'recommendation': ''
        }
        
        # Format supporting factors
        if len(top_negative) > 0:
            for _, row in top_negative.iterrows():
                explanation['key_factors']['supporting_approval'].append({
                    'factor': row['feature'],
                    'value': row['feature_value'],
                    'impact': f"Reduces default risk by {abs(row['shap_value']):.3f}"
                })
        
        if len(top_positive) > 0:
            for _, row in top_positive.iterrows():
                explanation['key_factors']['supporting_denial'].append({
                    'factor': row['feature'],
                    'value': row['feature_value'],
                    'impact': f"Increases default risk by {row['shap_value']:.3f}"
                })
        
        # Generate detailed explanation
        if decision == 'APPROVE':
            explanation['detailed_explanation'] = (
                f"The loan application has been APPROVED with confidence score {prediction_score:.3f}. "
                f"Key factors supporting approval include strong performance in areas such as "
                f"{', '.join([f['factor'] for f in explanation['key_factors']['supporting_approval'][:3]])}."
            )
            explanation['recommendation'] = "Proceed with loan approval following standard terms."
        else:
            explanation['detailed_explanation'] = (
                f"The loan application has been DENIED with confidence score {prediction_score:.3f}. "
                f"Key risk factors identified include "
                f"{', '.join([f['factor'] for f in explanation['key_factors']['supporting_denial'][:3]])}."
            )
            explanation['recommendation'] = "Consider manual review or request additional documentation."
        
        return explanation 