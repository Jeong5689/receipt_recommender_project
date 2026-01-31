import json
import os
from pathlib import Path

# 경로 설정
img_path = "data/receipts/train/images"
label_path = "data/receipts/train/labels/test.json"

print(f"--- 돋보기 진단 시작 ---")

# 1. 이미지 폴더 실제 내용 확인
if os.path.exists(img_path):
    files = os.listdir(img_path)
    print(f"폴더 내 파일 개수: {len(files)}")
    if files:
        print(f"폴더 내 첫 번째 파일명: [{files[0]}]")
else:
    print("❌ 이미지 폴더를 찾을 수 없습니다.")

# 2. JSON 파일 실제 내용 확인
if os.path.exists(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # images 키가 있는지 확인
    keys = list(data.keys())
    print(f"JSON 최상위 키 목록: {keys}")
    
    # 실제 데이터 샘플 추출
    target = data.get('images', data)
    if isinstance(target, dict):
        json_first_key = list(target.keys())[0]
        print(f"JSON 내 첫 번째 파일명 키: [{json_first_key}]")
        
        # 직접 비교 출력
        if files:
            print(f"\n--- 직접 비교 ---")
            print(f"폴더 파일명: {files[0]}")
            print(f"JSON 파일명: {json_first_key}")
            print(f"두 이름이 같은가? {files[0] == json_first_key}")
    else:
        print("❌ JSON 구조가 Dictionary가 아닙니다.")
else:
    print("❌ JSON 파일을 찾을 수 없습니다.")