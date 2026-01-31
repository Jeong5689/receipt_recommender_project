import torch
import torch.nn as nn
import cv2
import numpy as np
from lightning import LightningModule
from utils import calculate_ocr_metrics 

class DBNetOCR(nn.Module):
    def __init__(self):
        super().__init__()
        # 간단한 테스트용 레이어 (실제 프로젝트 시 ResNet 등으로 교체)
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1) 

    def forward(self, x):
        # DBNet의 결과물은 보통 0~1 사이의 확률 지도(Probability Map)입니다.
        binary_maps = torch.sigmoid(self.conv(x)) 
        return {"binary_maps": binary_maps}

class OCRLightningModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.l1_loss = nn.L1Loss()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_maps = batch['gt_maps']
        
        outputs = self(images)
        pred_maps = outputs['binary_maps']
        
        loss = self.l1_loss(pred_maps, gt_maps)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        gt_polygons = batch['polygons'] 
        
        outputs = self(images)
        pred_maps = outputs['binary_maps']
        
        # 1. 후처리 (중요!): Heatmap을 Polygon(좌표)으로 변환
        pred_polygons = self._post_process(pred_maps)
        
        # 2. utils에서 가져온 Shapely 기반 지표 계산
        metrics = calculate_ocr_metrics(pred_polygons, gt_polygons)
        
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # 모든 배치의 Precision, Recall 평균 계산
        avg_p = np.mean([x['precision'] for x in self.validation_step_outputs])
        avg_r = np.mean([x['recall'] for x in self.validation_step_outputs])
        
        hmean = 2 * (avg_p * avg_r) / (avg_p + avg_r + 1e-6)
        
        self.log("val_hmean", hmean, prog_bar=True)
        self.log("val_precision", avg_p)
        self.log("val_recall", avg_r)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def _post_process(self, pred_maps, thresh=0.3):
        """
        확률 지도를 다각형 좌표로 변환 (OpenCV 사용)
        """
        # 1. 텐서를 넘파이 배열로 변환
        pred_mask = pred_maps[0, 0].detach().cpu().numpy()
        
        # 2. 임계값(Threshold) 적용하여 이진화
        segmentation = (pred_mask > thresh).astype(np.uint8)
        
        # 3. 윤곽선(Contours) 찾기
        contours, _ = cv2.findContours(segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if len(contour) < 3: continue # 최소 삼각형 이상
            # 좌표 형태를 [[x1, y1], [x2, y2], ...] 리스트로 변환
            points = contour.reshape(-1, 2).tolist()
            boxes.append(points)
            
        return boxes