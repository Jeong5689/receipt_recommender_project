# 🧾 영수증 텍스트 탐지 및 추천 시스템 (Receipt Text Detector)

본 프로젝트는 딥러닝 모델 **DBNet**을 활용하여 영수증 이미지에서 텍스트 영역을 탐지하고, 추출된 데이터를 바탕으로 소비 패턴 분석 및 추천 서비스의 기반을 마련하는 시스템입니다.

## 🚀 프로젝트 목표
* 영수증 이미지 내의 메뉴, 가격 등 텍스트 라인 좌표의 자동 추출
* 저화질 및 복잡한 배경 노이즈 환경에서의 탐지 정밀도 개선
* 추출된 좌표 데이터를 구조화된 CSV 파일로 저장 및 관리

## 🏗️ 시스템 아키텍처 및 체계도
프로젝트는 크게 **이미지 전처리 -> 딥러닝 추론 -> 후처리 알고리즘(Heuristic)** 단계로 구성됩니다.

1. **Input**: 영수증 이미지 ($640 \times 640$)
2. **Model**: DBNet (Feature Pyramid Network 기반)
3. **Post-Processing**: Adaptive Thresholding & Grid Splitting
4. **Output**: 구조화된 좌표 데이터 (CSV) 및 시각화 결과물

## 📁 디렉토리 구조
```text
receipt_recommender_project/
├── src/
│   ├── model.py          # DBNet 모델 아키텍처 정의
│   ├── predict_batch.py  # 배치 추론 및 후처리 메인 로직
│   ├── dataset.py        # 데이터 로더 및 전처리
│   └── train.py          # 모델 학습 스크립트
├── all_detection_results.csv # 최종 추출 좌표 데이터 예시
├── requirements.txt      # 설치 라이브러리 목록
└── README.md             # 프로젝트 명세서

🛠️ 기술적 도전 및 해결 (Troubleshooting)
1. 배경 노이즈 및 오검출 제어
문제: 이미지 외곽의 바닥 무늬나 옷감 주름을 텍스트로 인식하는 현상 발생.

해결: Spatial ROI Constraint 기법을 적용하여 영수증이 위치한 중앙 영역(Central 60%) 외의 탐지 결과를 필터링함.

2. 거대 객체(Blob) 분할 문제
문제: 모델 학습도 한계로 인해 개별 텍스트 라인이 아닌 영수증 전체가 하나의 박스로 잡히는 현상 발생.

해결: Heuristic Grid Splitting 알고리즘을 도입. 검출된 거대 객체의 높이를 분석하여 영수증의 평균 행 높이 기준으로 좌표를 강제 분할함.

📊 결과 및 기대 효과
수율 개선: 초기 0~4개에 불과하던 텍스트 라인 검출 개수를 후처리를 통해 10~20개 수준으로 복구.

데이터 활용: 구축된 좌표 데이터셋을 활용하여 향후 OCR(EasyOCR 등) 연동 및 자동 지출 관리 서비스로 확장 가능.

⚙️ 시작하기

1. 필수 라이브러리 설치:
pip install -r requirements.txt

2. 추론 실행:
python src/predict_batch.py

### 💡 팁: GitHub에 적용하기
1. VS Code 탐색기에서 **`README.md`** 파일을 생성합니다.
2. 위 내용을 붙여넣고 저장(`Ctrl + S`)합니다.
3. 터미널에서 다음 명령어를 입력하여 업데이트합니다:
   ```bash
   git add README.md
   git commit -m "Add README file"
   git push origin main