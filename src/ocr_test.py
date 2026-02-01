import easyocr
import cv2
import pandas as pd
import os

def run_ocr_and_save():
    # [설정] 불러올 파일 및 저장할 경로
    CSV_INPUT = r"C:\Users\user\Documents\receipt_recommender_project\final_detection_v2.csv"
    IMAGE_DIR = r"C:\Users\user\Documents\receipt_recommender_project\data\receipts\val\images"
    # ⭐ 최종 저장될 위치
    OUTPUT_CSV = r"C:\Users\user\Documents\receipt_recommender_project\final_result_with_text.csv"
    
    print("⏳ OCR 엔진(EasyOCR)을 로드 중입니다...")
    reader = easyocr.Reader(['ko', 'en'])
    
    # 엑셀 로드
    df = pd.read_csv(CSV_INPUT)
    ocr_results = []

    print(f"🚀 총 {len(df)}개의 라인을 분석합니다. 잠시만 기다려주세요...")

    # 한 줄씩 읽으며 OCR 수행
    for i, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row['file_name'])
        image = cv2.imread(img_path)
        
        if image is None:
            ocr_results.append("")
            continue
            
        # 좌표 추출
        x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
        cropped = image[y:y+h, x:x+w]
        
        # 텍스트 인식
        result = reader.readtext(cropped, detail=0)
        text = result[0] if result else ""
        ocr_results.append(text)
        
        if i % 10 == 0: # 10줄마다 진행 상황 표시
            print(f"📊 진행률: {i}/{len(df)} 완료...")

    # 데이터프레임에 새로운 열 추가
    df['ocr_text'] = ocr_results
    
    # ⭐ 엑셀 저장
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 모든 작업이 완료되었습니다!")
    print(f"📁 저장된 위치: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_ocr_and_save()