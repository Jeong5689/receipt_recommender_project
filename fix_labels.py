import json
import os

# 경로 설정
img_dir = r"C:\Users\user\Documents\receipt_recommender_project\data\receipts\train\images"
old_json_path = r"C:\Users\user\Documents\receipt_recommender_project\data\receipts\train\labels\test.json"
new_json_path = r"C:\Users\user\Documents\receipt_recommender_project\data\receipts\train\labels\train_fixed.json"

def fix_labels():
    # 1. 실제 이미지 파일 목록 가져오기
    if not os.path.exists(img_dir):
        print(f"❌ 이미지 폴더를 찾을 수 없습니다: {img_dir}")
        return
    
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📂 발견된 이미지 개수: {len(img_files)}")

    # 2. 기존 JSON 로드
    with open(old_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    old_images = data.get('images', data)
    old_keys = list(old_images.keys())
    print(f"📋 JSON 내 데이터 개수: {len(old_keys)}")

    # 3. 강제 매칭 (이미지 파일명 기준으로 JSON 데이터 재구성)
    new_images_dict = {}
    
    # 이미지 파일과 JSON 데이터 중 더 적은 쪽의 개수만큼 매칭
    match_count = min(len(img_files), len(old_keys))
    
    for i in range(match_count):
        actual_file_name = img_files[i]  # 실제 폴더에 있는 이름
        json_data = old_images[old_keys[i]]  # JSON에 있던 좌표 등 정보
        
        new_images_dict[actual_file_name] = json_data

    # 4. 새 JSON 저장
    new_data = {"images": new_images_dict}
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 성공! {match_count}개의 데이터가 매칭된 '{new_json_path}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    fix_labels()