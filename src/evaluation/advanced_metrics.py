"""
Advanced Evaluation Metrics for SBA Loan Prediction
Implements cutting-edge 2025 evaluation techniques including gains/lift charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

class BusinessMetricsCalculator:
    """
    Calculate business-specific metrics for loan portfolio optimization
    """
    
    def __init__(self, profit_per_good_loan_pct: float = 0.05, 
                 loss_per_bad_loan_pct: float = 0.25):
        """
        Initialize with competition cost structure:
        - Profit per good loan = 5% of loan amount
        - Loss per bad loan = 25% of loan amount (5x cost ratio)
        """
        self.profit_per_good_loan_pct = profit_per_good_loan_pct
        self.loss_per_bad_loan_pct = loss_per_bad_loan_pct
    
    def calculate_portfolio_profit(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 loan_amounts: np.ndarray) -> Dict:
        """
        Calculate portfolio-level profit metrics
        
        Args:
            y_true: Actual default labels (1=default, 0=paid)
            y_pred: Predicted approval decisions (1=deny, 0=approve)
            loan_amounts: Loan disbursement amounts
            
        Returns:
            Dictionary with profit metrics
        """
        total_profit = 0
        approved_count = 0
        denied_count = 0
        
        # Calculate profit for each loan decision
        for i in range(len(y_true)):
            loan_amount = loan_amounts[i]
            actual_default = y_true[i]
            decision_approve = (y_pred[i] == 0)  # 0 = approve, 1 = deny
            
            if decision_approve:
                approved_count += 1
                if actual_default == 0:  # Good loan, paid in full
                    total_profit += self.profit_per_good_loan_pct * loan_amount
                else:  # Bad loan, defaulted
                    total_profit -= self.loss_per_bad_loan_pct * loan_amount
            else:
                denied_count += 1
                # No profit or loss from denied loans
        
        total_loans = len(y_true)
        approval_rate = approved_count / total_loans
        
        # Calculate default rate among approved loans
        approved_mask = (y_pred == 0)
        if approved_count > 0:
            default_rate_approved = y_true[approved_mask].mean()
        else:
            default_rate_approved = 0
        
        return {
            'total_profit': total_profit,
            'total_loans': total_loans,
            'approved_loans': approved_count,
            'denied_loans': denied_count,
            'approval_rate': approval_rate,
            'default_rate_approved': default_rate_approved,
            'profit_per_loan': total_profit / total_loans,
            'profit_per_approved': total_profit / max(approved_count, 1)
        }
    
    def calculate_threshold_profits(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  loan_amounts: np.ndarray, 
                                  thresholds: np.ndarray = None) -> pd.DataFrame:
        """
        Calculate profit for different probability thresholds
        
        Args:
            y_true: Actual default labels
            y_pred_proba: Predicted default probabilities
            loan_amounts: Loan amounts
            thresholds: Array of thresholds to test
            
        Returns:
            DataFrame with threshold analysis results
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = []
        
        for threshold in thresholds:
            # Convert probabilities to decisions (1=deny if prob > threshold)
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate business metrics
            metrics = self.calculate_portfolio_profit(y_true, y_pred, loan_amounts)
            metrics['threshold'] = threshold
            
            results.append(metrics)
        
        return pd.DataFrame(results)

class GainsLiftAnalyzer:
    """
    Create gains and lift charts for model evaluation
    """
    
    def __init__(self):
        self.gains_data = None
        self.lift_data = None
    
    def calculate_gains_lift(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           loan_amounts: np.ndarray, n_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate gains and lift data
        
        Args:
            y_true: Actual default labels
            y_pred_proba: Predicted default probabilities
            loan_amounts: Loan amounts
            n_bins: Number of bins for analysis
            
        Returns:
            Tuple of (gains_df, lift_df)
        """
        # Create dataframe with all data
        df = pd.DataFrame({
            'actual_default': y_true,
            'pred_proba': y_pred_proba,
            'loan_amount': loan_amounts
        })
        
        # Sort by predicted probability (descending - highest risk first)
        df = df.sort_values('pred_proba', ascending=False).reset_index(drop=True)
        
        # Create bins
        df['bin'] = pd.cut(range(len(df)), bins=n_bins, labels=False)
        
        # Calculate cumulative metrics
        total_defaults = df['actual_default'].sum()
        total_loans = len(df)
        overall_default_rate = total_defaults / total_loans
        
        gains_data = []
        lift_data = []
        
        for bin_num in range(n_bins):
            # Cumulative metrics up to this bin
            mask = df['bin'] <= bin_num
            cumulative_loans = mask.sum()
            cumulative_defaults = df[mask]['actual_default'].sum()
            
            # Gains calculation
            pct_loans = cumulative_loans / total_loans
            pct_defaults_captured = cumulative_defaults / total_defaults if total_defaults > 0 else 0
            
            # Lift calculation (this bin only)
            bin_mask = df['bin'] == bin_num
            if bin_mask.sum() > 0:
                bin_default_rate = df[bin_mask]['actual_default'].mean()
                lift = bin_default_rate / overall_default_rate if overall_default_rate > 0 else 1
            else:
                lift = 1
            
            gains_data.append({
                'bin': bin_num + 1,
                'pct_loans': pct_loans * 100,
                'pct_defaults_captured': pct_defaults_captured * 100,
                'cumulative_defaults': cumulative_defaults,
                'cumulative_loans': cumulative_loans
            })
            
            lift_data.append({
                'bin': bin_num + 1,
                'lift': lift,
                'default_rate': bin_default_rate if bin_mask.sum() > 0 else 0,
                'overall_default_rate': overall_default_rate
            })
        
        self.gains_data = pd.DataFrame(gains_data)
        self.lift_data = pd.DataFrame(lift_data)
        
        return self.gains_data, self.lift_data
    
    def plot_gains_chart(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive gains chart
        
        Args:
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        if self.gains_data is None:
            raise ValueError("Must run calculate_gains_lift first")
        
        fig = go.Figure()
        
        # Gains curve
        fig.add_trace(go.Scatter(
            x=self.gains_data['pct_loans'],
            y=self.gains_data['pct_defaults_captured'],
            mode='lines+markers',
            name='Gains Curve',
            line=dict(color='blue', width=3),
            hovertemplate='<b>Gains Chart</b><br>' +
                         'Loans Processed: %{x:.1f}%<br>' +
                         'Defaults Captured: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        # Baseline (random model)
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Random Model',
            line=dict(color='gray', dash='dash'),
            hovertemplate='<b>Random Model</b><br>' +
                         'Loans Processed: %{x:.1f}%<br>' +
                         'Defaults Captured: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Gains Chart - Model Performance in Capturing Defaults',
            xaxis_title='Percentage of Loans Processed (%)',
            yaxis_title='Percentage of Defaults Captured (%)',
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_lift_chart(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive lift chart
        
        Args:
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        if self.lift_data is None:
            raise ValueError("Must run calculate_gains_lift first")
        
        fig = go.Figure()
        
        # Lift curve
        fig.add_trace(go.Bar(
            x=self.lift_data['bin'],
            y=self.lift_data['lift'],
            name='Lift',
            marker_color='lightblue',
            hovertemplate='<b>Lift Chart</b><br>' +
                         'Bin: %{x}<br>' +
                         'Lift: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Baseline (lift = 1)
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Baseline (Random Model)")
        
        fig.update_layout(
            title='Lift Chart - Model Performance by Risk Decile',
            xaxis_title='Risk Decile (1 = Highest Risk)',
            yaxis_title='Lift (Model vs Random)',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

class AdvancedModelEvaluator:
    """
    Comprehensive model evaluation with business metrics
    """
    
    def __init__(self):
        self.business_calc = BusinessMetricsCalculator()
        self.gains_analyzer = GainsLiftAnalyzer()
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               loan_amounts: np.ndarray, 
                               threshold: float = 0.4) -> Dict:
        """
        Perform comprehensive model evaluation
        
        Args:
            y_true: Actual default labels
            y_pred_proba: Predicted default probabilities  
            loan_amounts: Loan amounts
            threshold: Decision threshold
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Convert probabilities to decisions
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Basic ML metrics
        auc_score = roc_auc_score(y_true, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Business metrics
        business_metrics = self.business_calc.calculate_portfolio_profit(y_true, y_pred, loan_amounts)
        
        # Threshold analysis
        threshold_analysis = self.business_calc.calculate_threshold_profits(
            y_true, y_pred_proba, loan_amounts
        )
        
        # Gains and lift analysis
        gains_df, lift_df = self.gains_analyzer.calculate_gains_lift(
            y_true, y_pred_proba, loan_amounts
        )
        
        return {
            'ml_metrics': {
                'auc': auc_score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            },
            'business_metrics': business_metrics,
            'threshold_analysis': threshold_analysis,
            'gains_data': gains_df,
            'lift_data': lift_df,
            'threshold_used': threshold
        } 