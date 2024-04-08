import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_sba_dataset():
    """Comprehensive exploration of the SBA dataset"""
    
    print("Loading SBA National Dataset...")
    df = pd.read_csv('../data/sba_national_data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total observations: {len(df):,}")
    print(f"Total variables: {len(df.columns)}")
    print()
    
    # Column names
    print("=== COLUMN NAMES ===")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    print()
    
    # Data types
    print("=== DATA TYPES ===")
    print(df.dtypes)
    print()
    
    # Target variable analysis
    print("=== TARGET VARIABLE (MIS_Status) ===")
    target_counts = df['MIS_Status'].value_counts()
    print(target_counts)
    print(f"Default rate (CHGOFF): {(df['MIS_Status'] == 'CHGOFF').mean():.3%}")
    print(f"Paid in full rate (P I F): {(df['MIS_Status'] == 'P I F').mean():.3%}")
    print()
    
    # Missing values analysis
    print("=== MISSING VALUES ANALYSIS ===")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print("Columns with missing values:")
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    print()
    
    # Numerical columns analysis
    print("=== NUMERICAL COLUMNS ANALYSIS ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical columns: {numerical_cols}")
    
    if numerical_cols:
        print("\nNumerical columns statistics:")
        print(df[numerical_cols].describe())
    print()
    
    # Categorical columns analysis
    print("=== CATEGORICAL COLUMNS ANALYSIS ===")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")
    
    for col in categorical_cols[:10]:  # Show first 10 categorical columns
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
        if unique_count <= 20:  # Show value counts for columns with few unique values
            print(f"  Values: {df[col].value_counts().head().to_dict()}")
    print()
    
    # Key business variables analysis
    print("=== KEY BUSINESS VARIABLES ===")
    
    # Loan amounts
    if 'DisbursementGross' in df.columns:
        # Clean the currency format
        df['DisbursementGross_Clean'] = df['DisbursementGross'].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
        df['DisbursementGross_Clean'] = pd.to_numeric(df['DisbursementGross_Clean'], errors='coerce')
        
        print("Loan Amount (DisbursementGross) Statistics:")
        print(f"  Mean: ${df['DisbursementGross_Clean'].mean():,.2f}")
        print(f"  Median: ${df['DisbursementGross_Clean'].median():,.2f}")
        print(f"  Min: ${df['DisbursementGross_Clean'].min():,.2f}")
        print(f"  Max: ${df['DisbursementGross_Clean'].max():,.2f}")
    
    # Term analysis
    if 'Term' in df.columns:
        print(f"\nLoan Term Statistics:")
        print(f"  Mean: {df['Term'].mean():.1f} months")
        print(f"  Most common terms: {df['Term'].value_counts().head().to_dict()}")
    
    # Employee count
    if 'NoEmp' in df.columns:
        print(f"\nNumber of Employees Statistics:")
        print(f"  Mean: {df['NoEmp'].mean():.1f}")
        print(f"  Median: {df['NoEmp'].median():.1f}")
        print(f"  Max: {df['NoEmp'].max()}")
    
    # State distribution
    if 'State' in df.columns:
        print(f"\nTop 10 States by loan count:")
        print(df['State'].value_counts().head(10).to_dict())
    
    # Industry (NAICS) distribution
    if 'NAICS' in df.columns:
        print(f"\nTop 10 Industries (NAICS) by loan count:")
        print(df['NAICS'].value_counts().head(10).to_dict())
    print()
    
    # Date analysis
    print("=== DATE ANALYSIS ===")
    if 'ApprovalFY' in df.columns:
        print("Approval Fiscal Year distribution:")
        # Convert to numeric for sorting, handle mixed types
        fy_counts = df['ApprovalFY'].value_counts()
        print(dict(list(fy_counts.items())[:10]))
    print()
    
    # Default analysis by key variables
    print("=== DEFAULT ANALYSIS BY KEY VARIABLES ===")
    
    # Default rate by business type
    if 'NewExist' in df.columns:
        default_by_newexist = df.groupby('NewExist')['MIS_Status'].apply(lambda x: (x == 'CHGOFF').mean())
        print("Default rate by Business Type (NewExist):")
        for idx, rate in default_by_newexist.items():
            business_type = "Existing" if idx == 1 else "New" if idx == 2 else f"Type_{idx}"
            print(f"  {business_type}: {rate:.3%}")
    
    # Default rate by urban/rural
    if 'UrbanRural' in df.columns:
        default_by_location = df.groupby('UrbanRural')['MIS_Status'].apply(lambda x: (x == 'CHGOFF').mean())
        print("\nDefault rate by Location Type (UrbanRural):")
        for idx, rate in default_by_location.items():
            location_type = "Undefined" if idx == 0 else "Urban" if idx == 1 else "Rural" if idx == 2 else f"Type_{idx}"
            print(f"  {location_type}: {rate:.3%}")
    
    # Default rate by LowDoc program
    if 'LowDoc' in df.columns:
        default_by_lowdoc = df.groupby('LowDoc')['MIS_Status'].apply(lambda x: (x == 'CHGOFF').mean())
        print("\nDefault rate by LowDoc Program:")
        for program, rate in default_by_lowdoc.items():
            print(f"  {program}: {rate:.3%}")
    print()
    
    # Sample records
    print("=== SAMPLE RECORDS ===")
    print("First 3 records:")
    print(df.head(3).to_string())
    print()
    
    print("=== DATA QUALITY ISSUES TO ADDRESS ===")
    issues = []
    
    # Check for currency formatting
    currency_cols = ['DisbursementGross', 'GrAppv', 'SBA_Appv', 'BalanceGross', 'ChgOffPrinGr']
    for col in currency_cols:
        if col in df.columns:
            sample_val = str(df[col].iloc[0])
            if '$' in sample_val or ',' in sample_val:
                issues.append(f"- {col} contains currency formatting ($ and commas)")
    
    # Check for date formatting
    date_cols = ['ApprovalDate', 'DisbursementDate', 'ChgOffDate']
    for col in date_cols:
        if col in df.columns and df[col].dtype == 'object':
            issues.append(f"- {col} needs date parsing")
    
    # Check for categorical encoding
    if 'NewExist' in df.columns:
        if df['NewExist'].dtype == 'object':
            issues.append("- NewExist needs numerical encoding")
    
    if 'UrbanRural' in df.columns:
        if df['UrbanRural'].dtype == 'object':
            issues.append("- UrbanRural needs numerical encoding")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
    else:
        print("No major data quality issues detected!")
    
    print("\n=== EXPLORATION COMPLETE ===")
    return df

if __name__ == "__main__":
    df = explore_sba_dataset() 