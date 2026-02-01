import pandas as pd
import os

def generate_recommendation():
    # 1. 데이터 로드
    REPORT_PATH = r"C:\Users\user\Documents\receipt_recommender_project\spending_report.csv"
    SAVE_PATH = r"C:\Users\user\Documents\receipt_recommender_project\final_recommendation.txt"
    
    if not os.path.exists(REPORT_PATH):
        print("❌ 분석 리포트가 없습니다. 이전 단계를 먼저 완료하세요.")
        return

    df = pd.read_csv(REPORT_PATH)
    df.columns = ['item', 'count']
    
    # 2. 가장 많이 구매한 항목(Top 1) 추출
    favorite_item = df.iloc[0]['item']
    buy_count = df.iloc[0]['count']
    
    # 3. 항목별 맞춤형 추천 로직 (간단한 규칙 기반)
    recommend_msg = ""
    
    if '도넛' in favorite_item or '티라미수' in favorite_item:
        user_type = "달콤한 디저트 마니아"
        suggestion = "근처 '랜디스 도넛'의 새로운 시즌 메뉴를 확인해보세요! 🍩"
    elif '커피' in favorite_item or '아메리카노' in favorite_item:
        user_type = "카페인 충전 전문가"
        suggestion = "자주 가시는 카페의 '원두 구독 서비스'를 이용하면 월 15%를 절약할 수 있어요! ☕"
    else:
        user_type = "알뜰한 스마트 컨슈머"
        suggestion = "비슷한 소비 패턴을 가진 분들이 자주 찾는 '가성비 맛집' 리스트를 보내드릴까요? 📋"

    # 4. 결과 구성
    final_text = f"""
    [ 🎁 스마트 영수증 개인화 리포트 ]
    ----------------------------------
    ▶ 고객님의 타입: {user_type}
    ▶ 최애 항목: {favorite_item} (총 {buy_count}회 발견)
    
    📢 추천 알림:
    "{suggestion}"
    ----------------------------------
    """
    
    print(final_text)
    
    # 5. 파일 저장
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"✅ 최종 추천 리포트 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    generate_recommendation()