import pandas as pd
import numpy as np

def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

def main():
    file_path = 'iii_die attatch/20000_BGTTV.xlsx'
    sheet_name = '통합_불량데이터_20000'
    print(f"Loading data from {file_path}, sheet: {sheet_name}...")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    da_cols = [
        'Lot_ID', 'WF_ID', 
        'DA_Equipment', 'DA_Head', 'DA_Epoxy_Batch', 
        'DA_Placement_Offset_um', 'DA_Bonding_Pressure_N', 'DA_Head_Speed_mm_s', 
        'DA_Crack_Defect'
    ]
    
    df_da = df[da_cols].copy()
    
    print("\n[1] 데이터 스키마 및 기본 정보")
    print(f"Total Rows: {len(df_da)}")
    print("Columns & Dtypes:")
    for col in df_da.columns:
        print(f" - {col}: {df_da[col].dtype}")
        
    print("\n[2] 결측치 및 중복값 확인")
    missing = df_da.isnull().sum()
    print("Missing Values:")
    for col, val in missing.items():
        if val > 0:
            print(f" - {col}: {val}")
    if missing.sum() == 0:
        print(" - No missing values detected.")
        
    duplicates = df_da.duplicated(subset=['Lot_ID', 'WF_ID']).sum()
    print(f"Duplicate Rows (based on Lot_ID, WF_ID): {duplicates}")

    print("\n[3] 수치형 변수 통계 요약 및 이상치(IQR 기준) 탐지")
    num_cols = ['DA_Placement_Offset_um', 'DA_Bonding_Pressure_N', 'DA_Head_Speed_mm_s']
    for col in num_cols:
        outlier_count = detect_outliers_iqr(df_da[col])
        print(f"--- {col} ---")
        print(f" Mean: {df_da[col].mean():.3f} / Std: {df_da[col].std():.3f}")
        print(f" Min:  {df_da[col].min():.3f} / Max: {df_da[col].max():.3f}")
        print(f" Outliers(IQR): {outlier_count} ({outlier_count/len(df_da)*100:.2f}%)")

    print("\n[4] 범주형 변수 편중(Skewness) 및 고유값 검사")
    cat_cols = ['DA_Equipment', 'DA_Head', 'DA_Epoxy_Batch']
    for col in cat_cols:
        counts = df_da[col].value_counts(dropna=False)
        print(f"--- {col} (Unique: {len(counts)}) ---")
        for val, cnt in counts.items():
            print(f" {val}: {cnt} ({cnt/len(df_da)*100:.2f}%)")
            
    print("\n[5] 종속 변수(Target) 분포")
    crack_counts = df_da['DA_Crack_Defect'].value_counts(dropna=False)
    for val, cnt in crack_counts.items():
        print(f" DA_Crack_Defect={val}: {cnt} ({cnt/len(df_da)*100:.2f}%)")

if __name__ == "__main__":
    main()
