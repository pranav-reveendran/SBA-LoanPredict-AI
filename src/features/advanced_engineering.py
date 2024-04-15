"""
Advanced Feature Engineering for SBA Loan Prediction System
Implements cutting-edge 2025 feature engineering techniques
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import polars as pl

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering with 2025 ML techniques
    """
    
    def __init__(self, target_column: str = 'Default'):
        self.target_column = target_column
        self.feature_importance_scores = {}
        self.temporal_features = []
        self.interaction_features = []
        self.risk_features = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated temporal features"""
        df_enhanced = df.copy()
        
        # Parse ApprovalFY to year
        if 'ApprovalFY' in df_enhanced.columns:
            df_enhanced['ApprovalYear'] = pd.to_numeric(df_enhanced['ApprovalFY'], errors='coerce')
            
            # Business cycle features
            df_enhanced['IsFinancialCrisis'] = (
                (df_enhanced['ApprovalYear'] >= 2007) & 
                (df_enhanced['ApprovalYear'] <= 2009)
            ).astype(int)
            
            df_enhanced['IsRecoveryPeriod'] = (
                (df_enhanced['ApprovalYear'] >= 2010) & 
                (df_enhanced['ApprovalYear'] <= 2012)
            ).astype(int)
        
        self.temporal_features = [col for col in df_enhanced.columns 
                                if col not in df.columns]
        
        return df_enhanced
    
    def create_geographic_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create geographic risk features based on economic indicators
        
        Args:
            df: Input dataframe with State column
            
        Returns:
            DataFrame with geographic risk features
        """
        df_enhanced = df.copy()
        
        # High-risk states based on historical default patterns
        high_risk_states = ['NV', 'FL', 'AZ', 'CA', 'MI', 'OH']
        df_enhanced['IsHighRiskState'] = (
            df_enhanced['State'].isin(high_risk_states)
        ).astype(int)
        
        # Economic regions
        west_coast = ['CA', 'OR', 'WA']
        northeast = ['NY', 'NJ', 'CT', 'MA', 'PA', 'RI', 'VT', 'NH', 'ME']
        southeast = ['FL', 'GA', 'SC', 'NC', 'VA', 'TN', 'KY', 'AL', 'MS', 'LA', 'AR']
        midwest = ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI']
        
        df_enhanced['IsWestCoast'] = df_enhanced['State'].isin(west_coast).astype(int)
        df_enhanced['IsNortheast'] = df_enhanced['State'].isin(northeast).astype(int)
        df_enhanced['IsSoutheast'] = df_enhanced['State'].isin(southeast).astype(int)
        df_enhanced['IsMidwest'] = df_enhanced['State'].isin(midwest).astype(int)
        
        # State-level loan volume (concentration risk)
        state_counts = df_enhanced['State'].value_counts()
        df_enhanced['StateLoanVolume'] = df_enhanced['State'].map(state_counts)
        df_enhanced['IsHighVolumeState'] = (
            df_enhanced['StateLoanVolume'] > state_counts.quantile(0.8)
        ).astype(int)
        
        return df_enhanced
    
    def create_industry_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create industry-specific risk features based on NAICS codes
        
        Args:
            df: Input dataframe with NAICS column
            
        Returns:
            DataFrame with industry risk features
        """
        df_enhanced = df.copy()
        
        # High-risk industries (based on historical default rates)
        high_risk_naics = [722, 623, 813, 532, 453, 448, 451]  # Restaurants, healthcare, etc.
        df_enhanced['IsHighRiskIndustry'] = (
            df_enhanced['NAICS'].astype(str).str[:3].astype(int, errors='ignore').isin(high_risk_naics)
        ).astype(int)
        
        # Essential vs non-essential businesses
        essential_naics = [621, 622, 445, 447, 492, 493]  # Healthcare, food, transportation
        df_enhanced['IsEssentialBusiness'] = (
            df_enhanced['NAICS'].astype(str).str[:3].astype(int, errors='ignore').isin(essential_naics)
        ).astype(int)
        
        # Industry concentration risk
        naics_counts = df_enhanced['NAICS'].value_counts()
        df_enhanced['IndustryLoanVolume'] = df_enhanced['NAICS'].map(naics_counts)
        df_enhanced['IsNicheIndustry'] = (
            df_enhanced['IndustryLoanVolume'] < naics_counts.quantile(0.2)
        ).astype(int)
        
        return df_enhanced
    
    def create_business_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated business risk indicators"""
        df_enhanced = df.copy()
        
        # Employee efficiency ratios
        if 'NoEmp' in df_enhanced.columns and 'DisbursementGross' in df_enhanced.columns:
            df_enhanced['LoanPerEmployee'] = (
                df_enhanced['DisbursementGross'] / np.maximum(df_enhanced['NoEmp'], 1)
            )
            df_enhanced['IsHighLoanPerEmployee'] = (
                df_enhanced['LoanPerEmployee'] > df_enhanced['LoanPerEmployee'].quantile(0.8)
            ).astype(int)
        
        # Job creation efficiency
        if 'CreateJob' in df_enhanced.columns and 'DisbursementGross' in df_enhanced.columns:
            df_enhanced['LoanPerJobCreated'] = (
                df_enhanced['DisbursementGross'] / np.maximum(df_enhanced['CreateJob'], 1)
            )
            df_enhanced['IsJobCreator'] = (df_enhanced['CreateJob'] > 0).astype(int)
        
        # SBA guarantee ratio
        if 'SBA_Appv' in df_enhanced.columns and 'GrAppv' in df_enhanced.columns:
            df_enhanced['SBAGuaranteeRatio'] = (
                df_enhanced['SBA_Appv'] / np.maximum(df_enhanced['GrAppv'], 1)
            )
            df_enhanced['IsHighGuarantee'] = (
                df_enhanced['SBAGuaranteeRatio'] > 0.75
            ).astype(int)
        
        # Loan utilization ratio
        if 'DisbursementGross' in df_enhanced.columns and 'GrAppv' in df_enhanced.columns:
            df_enhanced['LoanUtilizationRatio'] = (
                df_enhanced['DisbursementGross'] / np.maximum(df_enhanced['GrAppv'], 1)
            )
            df_enhanced['IsFullUtilization'] = (
                df_enhanced['LoanUtilizationRatio'] > 0.95
            ).astype(int)
        
        self.risk_features = [col for col in df_enhanced.columns 
                            if col not in df.columns]
        
        return df_enhanced
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create quantum-inspired interaction features
        
        Args:
            df: Input dataframe
            target_col: Target column for supervised interactions
            
        Returns:
            DataFrame with interaction features
        """
        df_enhanced = df.copy()
        
        # Key interaction pairs (based on business logic)
        interaction_pairs = [
            ('NoEmp', 'DisbursementGross'),
            ('Term', 'DisbursementGross'),
            ('NewExist', 'NoEmp'),
            ('UrbanRural', 'DisbursementGross'),
            ('SBAGuaranteeRatio', 'Term'),
            ('IsHighRiskState', 'IsHighRiskIndustry')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df_enhanced.columns and feat2 in df_enhanced.columns:
                # Multiplicative interaction
                df_enhanced[f'{feat1}_{feat2}_mult'] = (
                    df_enhanced[feat1] * df_enhanced[feat2]
                )
                
                # Ratio interaction (if both are positive)
                if df_enhanced[feat2].min() > 0:
                    df_enhanced[f'{feat1}_{feat2}_ratio'] = (
                        df_enhanced[feat1] / df_enhanced[feat2]
                    )
        
        # Quantum-inspired entanglement features
        if target_col and target_col in df_enhanced.columns:
            # Create features that capture quantum-like correlations
            numerical_cols = df_enhanced.select_dtypes(include=[np.number]).columns
            for col in numerical_cols[:5]:  # Limit to avoid explosion
                if col != target_col:
                    # Entangled feature: correlation with target
                    correlation = df_enhanced[col].corr(df_enhanced[target_col])
                    df_enhanced[f'{col}_target_entanglement'] = (
                        df_enhanced[col] * correlation
                    )
        
        self.interaction_features = [col for col in df_enhanced.columns 
                                   if col not in df.columns]
        
        return df_enhanced
    
    def apply_target_encoding(self, df: pd.DataFrame, 
                            categorical_cols: List[str],
                            target_col: str) -> pd.DataFrame:
        """
        Apply advanced target encoding with regularization
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical columns to encode
            target_col: Target column for encoding
            
        Returns:
            DataFrame with target-encoded features
        """
        df_enhanced = df.copy()
        
        for col in categorical_cols:
            if col in df_enhanced.columns:
                # Target encoding with smoothing
                encoder = TargetEncoder(smooth="auto", cv=5)
                
                # Fit and transform
                encoded_values = encoder.fit_transform(
                    df_enhanced[[col]], df_enhanced[target_col]
                )
                
                df_enhanced[f'{col}_target_encoded'] = encoded_values.flatten()
        
        return df_enhanced
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, 
                           k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select best features using mutual information and business logic
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            Tuple of (selected features dataframe, feature names list)
        """
        # Mutual information feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance scores
        feature_scores = selector.scores_
        self.feature_importance_scores = dict(zip(X.columns, feature_scores))
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering techniques"""
        print("Starting advanced feature engineering...")
        
        df_enhanced = df.copy()
        
        print("Creating temporal features...")
        df_enhanced = self.create_temporal_features(df_enhanced)
        
        print("Creating geographic risk features...")
        df_enhanced = self.create_geographic_risk_features(df_enhanced)
        
        print("Creating industry risk features...")
        df_enhanced = self.create_industry_risk_features(df_enhanced)
        
        print("Creating business risk features...")
        df_enhanced = self.create_business_risk_features(df_enhanced)
        
        print("Creating interaction features...")
        target_col = self.target_column if self.target_column in df_enhanced.columns else None
        df_enhanced = self.create_interaction_features(df_enhanced, target_col)
        
        # Apply target encoding for categorical variables
        if target_col:
            categorical_cols = ['State', 'BusinessSize', 'LoanSize']
            existing_cats = [col for col in categorical_cols if col in df_enhanced.columns]
            if existing_cats:
                print("Applying target encoding...")
                df_enhanced = self.apply_target_encoding(
                    df_enhanced, existing_cats, target_col
                )
        
        print(f"Feature engineering complete. Added {len(df_enhanced.columns) - len(df.columns)} new features.")
        
        return df_enhanced
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of engineered features
        
        Returns:
            Dictionary with feature engineering summary
        """
        return {
            'temporal_features': self.temporal_features,
            'interaction_features': self.interaction_features,
            'risk_features': self.risk_features,
            'feature_importance_scores': self.feature_importance_scores
        } 