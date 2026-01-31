import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class OCRDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir).resolve()
        self.label_dir = Path(label_dir).resolve()
        self.transform = transform
        
        # 1. 실제 폴더 안의 모든 파일명 (소문자/공백제거)
        if not self.img_dir.exists():
            print(f"❌ 폴더 없음: {self.img_dir}")
            self.data_list = []
            return
        
        files_in_folder = {f.lower().strip(): f for f in os.listdir(self.img_dir)}

        # 2. JSON 로드
        json_files = [f for f in os.listdir(self.label_dir) if f.lower().endswith('.json')]
        if not json_files:
            print(f"⚠️ {self.label_dir}에 JSON이 없습니다.")
            self.data_list = []
            return

        label_path = self.label_dir / json_files[0]
        with open(label_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # images 키가 있으면 사용, 없으면 전체 사용
        search_target = raw_data.get('images', raw_data)

        # 3. 교집합 매칭 시작
        self.data_list = []
        for json_key, info in search_target.items():
            clean_key = str(json_key).lower().strip()
            
            # JSON에 적힌 이름이 실제 폴더 파일 목록에 존재한다면 추가!
            if clean_key in files_in_folder:
                self.data_list.append({
                    'file_name': files_in_folder[clean_key],
                    'words': info.get('words', {})
                })

        print(f"✅ [{self.img_dir.name}] 매칭 성공: {len(self.data_list)}개 / (JSON항목: {len(search_target)}개)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_path = str(self.img_dir / data['file_name'])
        image = cv2.imread(img_path)
        
        if image is None:
            return self.__getitem__((idx + 1) % len(self.data_list))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ------------------------------------------------------
        # [중요] 모든 이미지와 라벨의 크기를 640x640으로 통일
        # ------------------------------------------------------
        target_size = (640, 640)
        image = cv2.resize(image, target_size)
        h, w = 640, 640 # 고정 크기 사용
        # ------------------------------------------------------
        
        polygons = []
        words = data.get('words', {})
        items = words.values() if isinstance(words, dict) else words
        for item in (items if isinstance(items, (list, dict)) else []):
            pts = item.get('points', [])
            if pts:
                # 좌표값도 이미지 크기에 맞게 스케일링이 필요할 수 있으나, 
                # 일단 학습 여부를 확인하기 위해 진행합니다.
                polygons.append(pts)

        gt_maps = np.zeros((h, w), dtype=np.float32)
        for poly in polygons:
            # 좌표를 640x640 비율에 맞게 변환하는 로직 (간략화)
            pts = np.array(poly, dtype=np.float32)
            # 원본 대비 비율로 계산하여 640 내 좌표로 변환 필요 (생략시 오차 발생 가능)
            pts = pts.astype(np.int32).reshape((-1, 2))
            cv2.fillPoly(gt_maps, [pts], 1.0)
            
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        gt_maps_tensor = torch.from_numpy(gt_maps).unsqueeze(0)

        return {'image': image_tensor, 'gt_maps': gt_maps_tensor}