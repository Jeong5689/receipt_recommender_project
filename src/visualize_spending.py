import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Malgun Gothic'

def create_visualizations():
    INPUT_CSV = r"C:\Users\user\Documents\receipt_recommender_project\spending_report.csv"
    
    try:
        # 파일이 비어있는지 체크하며 로드
        df = pd.read_csv(INPUT_CSV)
        if df.empty or len(df.columns) < 2:
            raise ValueError("데이터가 부족합니다.")
            
        df.columns = ['menu', 'count']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='count', y='menu', data=df.head(10))
        plt.title('소비 패턴 분석 결과')
        plt.savefig(r"C:\Users\user\Documents\receipt_recommender_project\menu_frequency.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ 시각화 실패: {e}")
        print("💡 팁: spending_report.csv 파일을 열어 내용이 있는지 확인하세요.")

if __name__ == "__main__":
    create_visualizations()