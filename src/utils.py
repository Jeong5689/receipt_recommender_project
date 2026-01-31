import numpy as np
from shapely.geometry import Polygon
import torch

def calculate_ocr_metrics(pred_polygons, gt_polygons, iou_threshold=0.5):
    """
    Shapely를 사용하여 Precision, Recall, Hmean을 계산합니다.
    """
    # 1. 데이터가 비어있는 경우 예외 처리
    if len(gt_polygons) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'hmean': 0.0}
    if len(pred_polygons) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'hmean': 0.0}

    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))
    
    # 2. 좌표 데이터를 Shapely Polygon 객체로 변환
    # p는 [[x1, y1], [x2, y2], ...] 형태여야 함
    gt_pols = [Polygon(p) for p in gt_polygons if len(p) >= 3]
    pred_pols = [Polygon(p) for p in pred_polygons if len(p) >= 3]

    # 3. 매칭 계산 (IoU 기반)
    for i, pred_p in enumerate(pred_pols):
        if not pred_p.is_valid: # 잘못된 다각형(교차 등) 처리
            pred_p = pred_p.buffer(0)
            
        for j, gt_p in enumerate(gt_pols):
            if gt_matched[j]:
                continue
            
            if not gt_p.is_valid:
                gt_p = gt_p.buffer(0)

            # 교집합 및 합집합 면적 계산
            try:
                intersection_area = pred_p.intersection(gt_p).area
                union_area = pred_p.union(gt_p).area
                iou = intersection_area / (union_area + 1e-6)
            except:
                iou = 0.0

            if iou >= iou_threshold:
                gt_matched[j] = 1
                pred_matched[i] = 1
                break

    # 4. 최종 지표 산출
    precision = np.sum(pred_matched) / len(pred_polygons) if len(pred_polygons) > 0 else 0
    recall = np.sum(gt_matched) / len(gt_polygons) if len(gt_polygons) > 0 else 0
    hmean = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'precision': precision,
        'recall': recall,
        'hmean': hmean
    }