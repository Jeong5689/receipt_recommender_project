import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from train import DBNetModule
import os
import csv  # CSV 저장을 위한 라이브러리 추가

def predict():
    # 1. 경로 설정
    checkpoint_path = r"C:\Users\user\Documents\receipt_recommender_project\checkpoints\final_model_v2.ckpt"
    input_image_path = r"C:\Users\user\Documents\receipt_recommender_project\data\receipts\val\images\drp.en_ko.in_house.selectstar_000007.jpg"
    output_img_path = r"C:\Users\user\Documents\receipt_recommender_project\result_v2.jpg"
    csv_path = r"C:\Users\user\Documents\receipt_recommender_project\detection_results.csv"

    # 2. 모델 로드 및 이미지 준비
    model = DBNetModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    original_img = cv2.imread(input_image_path)
    h_orig, w_orig = original_img.shape[:2]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)).unsqueeze(0)

    # 3. 추론
    with torch.no_grad():
        pred_map = model(img_tensor).cpu().numpy().squeeze()

    # 4. 후처리 및 박스 좌표 추출
    binary_map = (pred_map > 0.1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # CSV에 저장할 데이터 리스트 생성
    detection_data = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # 원본 크기로 복원
        rx, ry = int(x * w_orig / 640), int(y * h_orig / 640)
        rw, rh = int(w * w_orig / 640), int(h * h_orig / 640)
        
        # 이미지에 그리기
        cv2.rectangle(original_img, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
        
        # 리스트에 추가 (이미지명, 번호, x, y, 가로, 세로)
        detection_data.append([os.path.basename(input_image_path), i+1, rx, ry, rw, rh])

    # 5. 결과 저장 (이미지 및 CSV)
    cv2.imwrite(output_img_path, original_img)
    
    # CSV 파일 쓰기
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'box_id', 'x', 'y', 'width', 'height']) # 헤더
        writer.writerows(detection_data)

    print(f"✅ 결과 이미지 저장: {output_img_path}")
    print(f"📊 CSV 데이터 저장: {csv_path}")

if __name__ == "__main__":
    predict()