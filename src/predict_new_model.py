import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import csv
from pathlib import Path

# [1] 모델 구조 (새 모델의 아키텍처에 맞게 수정 필요 시 반영)
class DBNetModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 새로운 모델의 레이어 구성 (기존과 동일하다면 유지)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.final_conv = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        pred = nn.functional.interpolate(self.final_conv(x), size=(640, 640), mode='bilinear')
        return pred

def run_new_model_prediction():
    # 💡 경로 설정 (새로운 모델 파일 경로로 수정하세요)
    NEW_CHECKPOINT = r"C:\Users\user\Documents\receipt_recommender_project\checkpoints\final_model_v2.ckpt"
    INPUT_FOLDER = r"C:\Users\user\Documents\receipt_recommender_project\data\receipts\val\images"
    OUTPUT_FOLDER = r"C:\Users\user\Documents\receipt_recommender_project\output_results_new"
    CSV_PATH = r"C:\Users\user\Documents\receipt_recommender_project\final_detection_v2.csv"

# 파일 상단 30번째 줄 근처
    NEW_CHECKPOINT = r"C:\Users\user\Documents\receipt_recommender_project\checkpoints\final_model_v2.ckpt"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model = DBNetModule()

    # 모델 로드
    # ... (상단 코드 동일)
    model = DBNetModule()

    # 모델 가중치 로드 (상세 에러 출력 버전)
    try:
        if not os.path.exists(NEW_CHECKPOINT):
            print(f"❌ 파일이 존재하지 않습니다: {NEW_CHECKPOINT}")
            return

        ckpt = torch.load(NEW_CHECKPOINT, map_location='cpu', weights_only=False)
        
        # 1. State Dict 추출 (Lightning 혹은 일반 PyTorch 체크포인트 대응)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # 2. Key 이름 매칭 (접두어 'model.' 제거)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('model.', '') # model.layer1 -> layer1
            new_state_dict[name] = v
            
        # 3. 로드 시도
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 모델 로드 성공! (일부 누락 키 무시: {len(msg.missing_keys)}개)")
        
        model.eval()
    except Exception as e:
        import traceback
        print("❌ 상세 로드 실패 원인:")
        print(traceback.format_exc()) # 에러의 구체적인 위치와 이유를 다 보여줍니다.
        return

    all_results = []
    image_files = list(Path(INPUT_FOLDER).glob("*.j*"))

    for img_path in image_files:
        orig = cv2.imread(str(img_path))
        if orig is None: continue
        h_orig, w_orig = orig.shape[:2]

        # 전처리
        img = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (640, 640))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = model(img_tensor).numpy().squeeze()

        # [자동 임계값 전략] 모델마다 신호 세기가 다르므로 p_max 대비 40% 지점 탐색
        p_max = pred.max()
        thresh = p_max * 0.4
        binary = (pred > thresh).astype(np.uint8) * 255
        
        # 줄 사이를 떼어놓기 위한 수직 침식 예시
        kernel = np.ones((3, 1), np.uint8) # 세로로 긴 커널
        binary = cv2.erode(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # [수정된 로직] 박스가 너무 크면(예: 높이가 100px 이상) 영수증 전체로 간주하고 쪼갭니다.
            if h > 100: 
                num_lines = 12  # 영수증 한 장당 대략 12줄로 가정 (조정 가능)
                line_h = h // num_lines
                
                for j in range(num_lines):
                    split_y = y + (j * line_h)
                    # 원본 이미지 좌표로 변환
                    margin = int(w_orig * 0.2) 
                    rx = margin
                    rw = w_orig - (margin * 2)
                    ry = int(split_y * h_orig / 640)
                    rh = int(line_h * h_orig / 640)
                    
                    all_results.append([img_path.name, count + 1, rx, ry, rw, rh])
                    cv2.rectangle(orig, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                    count += 1
            
            # 일반적인 크기의 박스는 그대로 저장
            elif h > 10 and w > 10:
                rx, ry = int(x * w_orig / 640), int(y * h_orig / 640)
                rw, rh = int(w * w_orig / 640), int(h * h_orig / 640)
                all_results.append([img_path.name, count + 1, rx, ry, rw, rh])
                cv2.rectangle(orig, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                count += 1

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, img_path.name), orig)
        print(f"📄 {img_path.name} | 새 모델 검출 완료: {count}개")

    # 결과 저장
    with open(CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'box_id', 'x', 'y', 'width', 'height'])
        writer.writerows(all_results)
    print(f"🎉 작업 완료! CSV 확인: {CSV_PATH}")

if __name__ == "__main__":
    run_new_model_prediction()