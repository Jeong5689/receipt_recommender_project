import pandas as pd
import os

def analyze_receipt_data():
    INPUT_CSV = r"C:\Users\user\Documents\receipt_recommender_project\final_result_with_text.csv"
    OUTPUT_REPORT = r"C:\Users\user\Documents\receipt_recommender_project\spending_report.csv"
    
    if not os.path.exists(INPUT_CSV):
        print("❌ OCR 결과 파일이 없습니다.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 공백 제거 및 짧은 단어 필터링 완화 (2글자 이상 -> 1글자 이상)
    df['ocr_text'] = df['ocr_text'].fillna('').astype(str).str.strip()
    filtered_df = df[df['ocr_text'].str.len() >= 1]

    if filtered_df.empty:
        print("⚠️ 인식된 텍스트가 너무 적습니다. 임시 데이터를 생성합니다.")
        menu_counts = pd.Series({'데이터 없음': 1})
    else:
        # 상위 빈도 추출
        menu_counts = filtered_df['ocr_text'].value_counts()

    # 결과 저장
    menu_counts.to_csv(OUTPUT_REPORT, header=['count'])
    print(f"✅ 분석 완료! 파일 확인: {OUTPUT_REPORT}")

if __name__ == "__main__":
    analyze_receipt_data()