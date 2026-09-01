import pandas as pd
import numpy as np
import scipy.stats as stats

def main():
    file_path = 'iii_die attatch/20000_BGTTV.xlsx'
    sheet_name = '통합_불량데이터_20000'
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    da_cols = [
        'DA_Equipment', 'DA_Head', 'DA_Epoxy_Batch', 
        'DA_Placement_Offset_um', 'DA_Bonding_Pressure_N', 'DA_Head_Speed_mm_s', 
        'DA_Crack_Defect'
    ]
    df_da = df[da_cols].copy()
    
    print("=== [EDA] 1. 수치형 공정 조건과 크랙 불량(Crack Defect)의 관계 ===")
    for col in ['DA_Bonding_Pressure_N', 'DA_Head_Speed_mm_s', 'DA_Placement_Offset_um']:
        mean_0 = df_da[df_da['DA_Crack_Defect'] == 0][col].mean()
        mean_1 = df_da[df_da['DA_Crack_Defect'] == 1][col].mean()
        t_stat, p_val = stats.ttest_ind(
            df_da[df_da['DA_Crack_Defect'] == 0][col],
            df_da[df_da['DA_Crack_Defect'] == 1][col],
            equal_var=False
        )
        print(f"[{col}]")
        print(f"  - 정상(0) 평균: {mean_0:.2f} | 크랙(1) 평균: {mean_1:.2f}")
        print(f"  - T-test p-value: {p_val:.4e} (p<0.05 이면 통계적 유의미)")
        
    print("\n=== [EDA] 2. 범주형 요인별 크랙 불량률(%) ===")
    for col in ['DA_Equipment', 'DA_Head', 'DA_Epoxy_Batch']:
        defect_rates = df_da.groupby(col)['DA_Crack_Defect'].mean() * 100
        print(f"[{col}]")
        for idx, rate in defect_rates.items():
            print(f"  - {idx}: {rate:.2f}%")
            
    print("\n=== [EDA] 3. 대안 가설 검증: 특정 장비/헤드에서 압력이 비정상적으로 설정되는가? ===")
    # 크랙 불량이 압력 때문이라면, 장비별로 압력 설정에 차이가 있는지 확인
    eq_pressure = df_da.groupby('DA_Equipment')['DA_Bonding_Pressure_N'].mean()
    head_pressure = df_da.groupby('DA_Head')['DA_Bonding_Pressure_N'].mean()
    print("[장비별 평균 본딩 압력]")
    print(eq_pressure.to_string())
    print("[헤드별 평균 본딩 압력]")
    print(head_pressure.to_string())
    
    # ANOVA test for Equipment vs Pressure
    groups = [group['DA_Bonding_Pressure_N'].values for name, group in df_da.groupby('DA_Equipment')]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n장비별 본딩 압력 차이 ANOVA p-value: {p_val:.4e}")

    print("\n=== [EDA] 4. 교호 작용: 장비(Equipment)와 헤드(Head) 조합별 불량률 ===")
    interaction = df_da.groupby(['DA_Equipment', 'DA_Head'])['DA_Crack_Defect'].agg(['mean', 'count'])
    interaction['mean'] = interaction['mean'] * 100
    interaction.rename(columns={'mean': 'Defect_Rate(%)', 'count': 'Sample_Count'}, inplace=True)
    print(interaction.to_string())

if __name__ == "__main__":
    main()
