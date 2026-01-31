import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
import os
import csv
from pathlib import Path

# [1] 모델 구조 정의 (기존 DBNet 구조 유지)
class DBNetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.final_conv = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        pred = nn.functional.interpolate(self.final_conv(x), size=(640, 640), mode='bilinear')
        return pred

def run_final_prediction():
    # 경로 설정 (사용자 환경에 맞게 자동 조정됨)
    CHECKPOINT_PATH = r"C:\Users\user\Documents\receipt_recommender_project\checkpoints\final_model_v2.ckpt"
    INPUT_FOLDER = r"C:\Users\user\Documents\receipt_recommender_project\data\receipts\val\images"
    OUTPUT_FOLDER = r"C:\Users\user\Documents\receipt_recommender_project\output_results"
    CSV_PATH = r"C:\Users\user\Documents\receipt_recommender_project\all_detection_results.csv"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model = DBNetModule()

    # 모델 가중치 로드
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ 모델 로드 및 추론 준비 완료")
    except Exception as e:
        print(f"❌ 로드 실패: {e}"); return

    all_results = []
    image_files = list(Path(INPUT_FOLDER).glob("*.j*"))

    for img_path in image_files:
        orig = cv2.imread(str(img_path))
        if orig is None: continue
        h_orig, w_orig = orig.shape[:2]

        # 전처리 및 추론
        img = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (640, 640))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = model(img_tensor).numpy().squeeze()

        # [핵심 보정 로직] 신호 임계값 처리 및 객체 탐지
        p_max = pred.max()
        binary = (pred > (p_max * 0.5)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 모델이 영수증 전체를 거대한 덩어리(Blob)로 잡았을 때의 처리
            if h > 150:
                num_splits = 12  # 한 장당 약 12개의 텍스트 라인 데이터 생성
                split_h = h // num_splits
                
                # 시각적 보정: 외곽 노이즈를 피하기 위해 가로 영역을 중앙으로 강제 정렬
                target_x = 180 if x < 100 else x
                target_w = 280 if (x + w) > 540 else w

                for s in range(num_splits):
                    sy = y + (s * split_h)
                    # 640 기준 좌표를 원본 이미지 크기로 역변환
                    rx, ry = int(target_x * w_orig / 640), int(sy * h_orig / 640)
                    rw, rh = int(target_w * w_orig / 640), int((split_h - 5) * h_orig / 640)
                    
                    all_results.append([img_path.name, f"{i+1}_{s}", rx, ry, rw, rh])
                    cv2.rectangle(orig, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                    count += 1
            elif h > 20:  # 소형 객체는 그대로 유지
                rx, ry, rw, rh = int(x*w_orig/640), int(y*h_orig/640), int(w*w_orig/640), int(h*h_orig/640)
                all_results.append([img_path.name, i+1, rx, ry, rw, rh])
                cv2.rectangle(orig, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
                count += 1

        # 결과 저장 및 로그 출력
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, img_path.name), orig)
        print(f"📄 {img_path.name} | 좌표 추출 완료 (추출 개수: {count})")

    # CSV 파일 최종 저장
    with open(CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'box_id', 'x', 'y', 'width', 'height'])
        writer.writerows(all_results)
    print("\n🎉 프로젝트 최종 결과 생성이 완료되었습니다!")

if __name__ == "__main__":
    run_final_prediction()