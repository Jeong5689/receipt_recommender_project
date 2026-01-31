from ocr_model import DBNetOCR
import torch
from torchvision import transforms
from PIL import Image

class OCREngine:
    def __init__(self):
        self.model = DBNetOCR()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def extract_items(self, image_path, gt=None):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)  # (1, C, H, W)
        with torch.no_grad():
            result = self.model(x, return_loss=False, gt=gt)
        # 더미로 text extraction -> 실제 CRNN 등 연결 가능
        return ["상품A", "상품B"]