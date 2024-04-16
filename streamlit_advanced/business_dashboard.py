"""
Advanced Business Intelligence Dashboard for SBA Loan Prediction System
Implements cutting-edge 2025 Streamlit features with real-time analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set page config
st.set_page_config(
    page_title="SBA Loan Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .denied {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class APIClient:
    """Client for interacting with Flask API"""
    
    def __init__(self, base_url="http://127.0.0.1:5001"):
        self.base_url = base_url
    
    def test_connection(self):
        """Test API connection"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_models(self):
        """Get available models"""
        try:
            response = requests.get(f"{self.base_url}/api/models", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def predict_single(self, loan_data):
        """Make single prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=loan_data,
                timeout=10
            )
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def predict_batch(self, loans_data):
        """Make batch predictions"""
        try:
            response = requests.post(
                f"{self.base_url}/api/predict/batch",
                json={"loans": loans_data},
                timeout=30
            )
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def get_analytics(self):
        """Get model analytics"""
        try:
            response = requests.get(f"{self.base_url}/api/analytics", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def get_features(self):
        """Get feature information"""
        try:
            response = requests.get(f"{self.base_url}/api/features", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

def create_gauge_chart(value, title, max_value=1.0, color_threshold=0.5):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': color_threshold},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, color_threshold], 'color': "lightgray"},
                {'range': [color_threshold, max_value], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': color_threshold
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_risk_distribution_chart(predictions):
    """Create risk distribution chart"""
    if not predictions:
        return go.Figure()
    
    # Extract risk levels
    risk_levels = [p.get('risk_level', 'Unknown') for p in predictions]
    risk_counts = pd.Series(risk_levels).value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=['green', 'orange', 'red'][:len(risk_counts)]
        )
    ])
    
    fig.update_layout(
        title="Risk Level Distribution",
        xaxis_title="Risk Level",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_profit_analysis_chart(predictions):
    """Create profit analysis chart"""
    if not predictions:
        return go.Figure()
    
    # Calculate cumulative expected value
    expected_values = [p.get('expected_value', 0) for p in predictions]
    cumulative_values = np.cumsum(expected_values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_values) + 1)),
        y=cumulative_values,
        mode='lines+markers',
        name='Cumulative Expected Value',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title="Cumulative Portfolio Expected Value",
        xaxis_title="Loan Number",
        yaxis_title="Expected Value ($)",
        height=400
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Initialize API client
    api_client = APIClient()
    
    # Header
    st.markdown('<h1 class="main-header">🏦 SBA Loan Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Dashboard Controls")
    
    # API Status
    st.sidebar.subheader("🔌 API Status")
    if api_client.test_connection():
        st.sidebar.success("✅ Connected to ML API")
        api_status = True
    else:
        st.sidebar.error("❌ Cannot connect to ML API")
        st.sidebar.info("Please ensure the Flask API is running on port 5001")
        api_status = False
    
    # Navigation
    page = st.sidebar.selectbox(
        "📍 Navigate",
        ["🏠 Overview", "🎯 Single Prediction", "📊 Batch Analysis", "📈 Model Analytics", "⚙️ Business Intelligence"]
    )
    
    if page == "🏠 Overview":
        show_overview_page(api_client, api_status)
    elif page == "🎯 Single Prediction":
        show_single_prediction_page(api_client, api_status)
    elif page == "📊 Batch Analysis":
        show_batch_analysis_page(api_client, api_status)
    elif page == "📈 Model Analytics":
        show_model_analytics_page(api_client, api_status)
    elif page == "⚙️ Business Intelligence":
        show_business_intelligence_page(api_client, api_status)

def show_overview_page(api_client, api_status):
    """Show overview page"""
    st.header("📋 System Overview")
    
    if api_status:
        # Get model information
        models = api_client.get_models()
        analytics = api_client.get_analytics()
        
        if models and analytics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f'<div class="metric-container"><h3>🤖 Model</h3><p>{models.get("model_name", "Unknown")}</p></div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<div class="metric-container"><h3>🎯 Accuracy</h3><p>{analytics.get("model_summary", {}).get("accuracy", 0)*100:.1f}%</p></div>',
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f'<div class="metric-container"><h3>💰 Expected Profit</h3><p>${analytics.get("model_summary", {}).get("expected_profit", 0):,.0f}</p></div>',
                    unsafe_allow_html=True
                )
            
            with col4:
                st.markdown(
                    f'<div class="metric-container"><h3>✅ Approval Rate</h3><p>{analytics.get("model_summary", {}).get("approval_rate", 0)*100:.1f}%</p></div>',
                    unsafe_allow_html=True
                )
            
            # Show model performance metrics
            st.subheader("📊 Model Performance")
            
            if "model_summary" in analytics:
                summary = analytics["model_summary"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # AUC Gauge
                    auc_fig = create_gauge_chart(
                        summary.get("auc", 0),
                        "AUC Score",
                        max_value=1.0,
                        color_threshold=0.8
                    )
                    st.plotly_chart(auc_fig, use_container_width=True)
                
                with col2:
                    # Precision Gauge
                    precision_fig = create_gauge_chart(
                        summary.get("precision", 0),
                        "Precision",
                        max_value=1.0,
                        color_threshold=0.7
                    )
                    st.plotly_chart(precision_fig, use_container_width=True)
        else:
            st.warning("⚠️ Unable to load model information")
    else:
        st.error("🚫 API connection required for overview")

def show_single_prediction_page(api_client, api_status):
    """Show single prediction page"""
    st.header("🎯 Single Loan Prediction")
    
    if not api_status:
        st.error("🚫 API connection required for predictions")
        return
    
    # Get feature information
    features_info = api_client.get_features()
    
    if not features_info:
        st.error("❌ Unable to load feature information")
        return
    
    st.subheader("📝 Loan Application Details")
    
    # Create input form
    with st.form("loan_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            term = st.number_input("📅 Loan Term (months)", min_value=1, max_value=600, value=120)
            no_emp = st.number_input("👥 Number of Employees", min_value=0, max_value=1000, value=10)
            create_job = st.number_input("➕ Jobs Created", min_value=0, max_value=100, value=0)
            retained_job = st.number_input("🔄 Jobs Retained", min_value=0, max_value=100, value=0)
            disbursement_gross = st.number_input("💰 Loan Amount ($)", min_value=1000, max_value=10000000, value=250000)
            gr_appv = st.number_input("✅ Gross Approved ($)", min_value=1000, max_value=10000000, value=275000)
        
        with col2:
            sba_appv = st.number_input("🏛️ SBA Guaranteed ($)", min_value=0, max_value=10000000, value=200000)
            new_exist = st.selectbox("🏢 Business Type", [1, 2], format_func=lambda x: "Existing" if x == 1 else "New", index=0)
            urban_rural = st.selectbox("🏙️ Location", [0, 1, 2], format_func=lambda x: "Undefined" if x == 0 else "Urban" if x == 1 else "Rural", index=1)
            revline_cr = st.selectbox("🔄 Revolving Credit", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
            low_doc = st.selectbox("📄 Low Documentation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=1)
            naics = st.number_input("🏭 NAICS Industry Code", min_value=100, max_value=999999, value=561)
        
        submit_button = st.form_submit_button("🔮 Predict Default Risk")
    
    if submit_button:
        # Prepare data for API
        loan_data = {
            "Term": term,
            "NoEmp": no_emp,
            "CreateJob": create_job,
            "RetainedJob": retained_job,
            "DisbursementGross": disbursement_gross,
            "GrAppv": gr_appv,
            "SBA_Appv": sba_appv,
            "NewExist": new_exist,
            "UrbanRural": urban_rural,
            "RevLineCr": revline_cr,
            "LowDoc": low_doc,
            "NAICS": naics
        }
        
        # Make prediction
        with st.spinner("🔄 Making prediction..."):
            result = api_client.predict_single(loan_data)
        
        if result and "prediction" in result:
            prediction = result["prediction"]
            
            # Display result
            st.subheader("🎯 Prediction Results")
            
            # Main prediction result
            if prediction["recommend_approval"]:
                st.markdown(
                    f'<div class="prediction-result approved">✅ RECOMMENDED FOR APPROVAL</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-result denied">❌ RECOMMENDED FOR DENIAL</div>',
                    unsafe_allow_html=True
                )
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📊 Default Probability",
                    f"{prediction['default_probability']*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "⚠️ Risk Level",
                    prediction['risk_level']
                )
            
            with col3:
                st.metric(
                    "💰 Expected Value",
                    f"${prediction['expected_value']:,.0f}"
                )
            
            # Risk gauge chart
            risk_gauge = create_gauge_chart(
                prediction['default_probability'],
                "Default Risk",
                max_value=1.0,
                color_threshold=prediction['threshold_used']
            )
            st.plotly_chart(risk_gauge, use_container_width=True)
            
        else:
            st.error("❌ Prediction failed. Please try again.")

def show_batch_analysis_page(api_client, api_status):
    """Show batch analysis page"""
    st.header("📊 Batch Loan Analysis")
    
    if not api_status:
        st.error("🚫 API connection required for batch analysis")
        return
    
    st.subheader("📤 Upload Loan Data")
    
    # Sample data download
    if st.button("📥 Download Sample CSV Template"):
        sample_data = pd.DataFrame({
            'Term': [120, 84, 60],
            'NoEmp': [10, 5, 15],
            'CreateJob': [0, 2, 1],
            'RetainedJob': [0, 3, 5],
            'DisbursementGross': [250000, 150000, 350000],
            'GrAppv': [275000, 165000, 385000],
            'SBA_Appv': [200000, 120000, 280000],
            'NewExist': [1, 2, 1],
            'UrbanRural': [1, 1, 2],
            'RevLineCr': [0, 0, 1],
            'LowDoc': [1, 1, 0],
            'NAICS': [561, 722, 541]
        })
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="sample_loans.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} loan applications")
            
            # Show preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Analysis"):
                # Convert to list of dictionaries
                loans_data = df.to_dict('records')
                
                with st.spinner("🔄 Processing batch predictions..."):
                    result = api_client.predict_batch(loans_data)
                
                if result and "predictions" in result:
                    predictions = result["predictions"]
                    summary = result["summary"]
                    
                    # Summary metrics
                    st.subheader("📈 Batch Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Total Loans", summary["total_loans"])
                    
                    with col2:
                        st.metric("✅ Approved", summary["approved_loans"])
                    
                    with col3:
                        st.metric("📋 Approval Rate", f"{summary['approval_rate']*100:.1f}%")
                    
                    with col4:
                        st.metric("💰 Total Expected Value", f"${summary['total_expected_value']:,.0f}")
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_chart = create_risk_distribution_chart(predictions)
                        st.plotly_chart(risk_chart, use_container_width=True)
                    
                    with col2:
                        profit_chart = create_profit_analysis_chart(predictions)
                        st.plotly_chart(profit_chart, use_container_width=True)
                    
                    # Detailed results
                    st.subheader("📋 Detailed Results")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(predictions)
                    results_df["Recommendation"] = results_df["recommend_approval"].map({True: "APPROVE", False: "DENY"})
                    
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ Batch analysis failed. Please try again.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def show_model_analytics_page(api_client, api_status):
    """Show model analytics page"""
    st.header("📈 Model Analytics & Performance")
    
    if not api_status:
        st.error("🚫 API connection required for analytics")
        return
    
    analytics = api_client.get_analytics()
    
    if analytics and "model_summary" in analytics:
        summary = analytics["model_summary"]
        
        # Model Information
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Model Name:** {summary.get('best_model', 'Unknown')}")
            st.info(f"**Optimal Threshold:** {summary.get('optimal_threshold', 0):.3f}")
            st.info(f"**Training Date:** {analytics.get('timestamp', 'Unknown')}")
        
        with col2:
            st.info(f"**Test Samples:** {summary.get('test_samples', 0):,}")
            st.info(f"**Features:** {summary.get('n_features', 0)}")
            st.info(f"**Cross-validation:** {summary.get('cv_score', 0):.3f}")
        
        # Performance Metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auc_gauge = create_gauge_chart(summary.get("auc", 0), "AUC Score", 1.0, 0.8)
            st.plotly_chart(auc_gauge, use_container_width=True)
        
        with col2:
            acc_gauge = create_gauge_chart(summary.get("accuracy", 0), "Accuracy", 1.0, 0.8)
            st.plotly_chart(acc_gauge, use_container_width=True)
        
        with col3:
            prec_gauge = create_gauge_chart(summary.get("precision", 0), "Precision", 1.0, 0.7)
            st.plotly_chart(prec_gauge, use_container_width=True)
        
        with col4:
            rec_gauge = create_gauge_chart(summary.get("recall", 0), "Recall", 1.0, 0.7)
            st.plotly_chart(rec_gauge, use_container_width=True)
        
        # Business Metrics
        st.subheader("💼 Business Impact")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💰 Expected Profit",
                f"${summary.get('expected_profit', 0):,.0f}"
            )
        
        with col2:
            st.metric(
                "✅ Approval Rate",
                f"{summary.get('approval_rate', 0)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "📊 Portfolio Value",
                f"${summary.get('total_loan_amount', 0):,.0f}"
            )
        
    else:
        st.error("❌ Unable to load model analytics")

def show_business_intelligence_page(api_client, api_status):
    """Show business intelligence page"""
    st.header("⚙️ Business Intelligence & Optimization")
    
    st.subheader("🎯 Threshold Optimization")
    st.info("""
    **Current Business Model:**
    - 5% profit margin on performing loans
    - 25% loss on defaulted loans (5:1 cost ratio)
    - Optimization target: Maximum portfolio profit
    """)
    
    # Real-time metrics (simulated)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Current portfolio status
        st.metric(
            "📊 Current Portfolio",
            "879,164 loans",
            delta="↗️ +2.3% vs last month"
        )
    
    with col2:
        # Default rate
        st.metric(
            "⚠️ Default Rate",
            "17.5%",
            delta="↘️ -0.8% vs target"
        )
    
    with col3:
        # Profit optimization
        st.metric(
            "💰 Profit Potential",
            "$1.33B",
            delta="↗️ +12% optimized"
        )
    
    # Economic indicators
    st.subheader("📈 Economic Indicators Impact")
    
    # Create sample economic data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    economic_data = pd.DataFrame({
        'Date': dates,
        'GDP_Growth': np.random.normal(2.5, 0.5, len(dates)),
        'Unemployment': np.random.normal(4.0, 0.8, len(dates)),
        'Default_Rate': np.random.normal(17.5, 2.0, len(dates))
    })
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GDP Growth (%)', 'Unemployment Rate (%)', 'Default Rate Trend (%)', 'Economic Impact Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # GDP Growth
    fig.add_trace(
        go.Scatter(x=economic_data['Date'], y=economic_data['GDP_Growth'], 
                  mode='lines', name='GDP Growth', line=dict(color='green')),
        row=1, col=1
    )
    
    # Unemployment
    fig.add_trace(
        go.Scatter(x=economic_data['Date'], y=economic_data['Unemployment'], 
                  mode='lines', name='Unemployment', line=dict(color='orange')),
        row=1, col=2
    )
    
    # Default Rate
    fig.add_trace(
        go.Scatter(x=economic_data['Date'], y=economic_data['Default_Rate'], 
                  mode='lines', name='Default Rate', line=dict(color='red')),
        row=2, col=1
    )
    
    # Economic Impact Score (composite)
    impact_score = 100 - (economic_data['Default_Rate'] - economic_data['GDP_Growth'] + economic_data['Unemployment'])
    fig.add_trace(
        go.Scatter(x=economic_data['Date'], y=impact_score, 
                  mode='lines', name='Impact Score', line=dict(color='blue')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Economic Indicators Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("🎯 AI-Powered Recommendations")
    
    recommendations = [
        "🔴 **High Priority**: Consider tightening lending criteria for NAICS codes 72 (Accommodation) due to elevated default rates",
        "🟡 **Medium Priority**: Implement dynamic pricing based on regional economic indicators",
        "🟢 **Low Priority**: Expand lending in technology sector (NAICS 54) showing strong performance",
        "📊 **Analytics**: Current model confidence is 94.5% - consider retraining in Q2 2024",
        "💰 **Profit Optimization**: Adjust threshold to 0.18 for +$85M additional profit potential"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

if __name__ == "__main__":
    main()
