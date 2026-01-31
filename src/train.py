import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L
import torchvision.transforms as transforms

# --- 모델 구조 ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.conv(x))

class DBNetOCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer2 = ResidualBlock(32, 32)
        self.layer3 = ResidualBlock(32, 32)
        self.final_conv = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.final_conv(x)

class DBNetModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DBNetOCR()
        self.criterion = nn.BCELoss()
    def forward(self, x): return self.model(x)
    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch['image'])
        loss = self.criterion(y_hat, batch['gt_maps'])
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# --- 데이터셋 ---
class OCRDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir, self.label_dir, self.transform = Path(img_dir), Path(label_dir), transform
        self.data_list = []
        with open(self.label_dir / "train_fixed.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        for file_name, content in data.get('images', data).items():
            self.data_list.append({'file_name': file_name, 'words': content.get('words', [])})
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = cv2.imread(str(self.img_dir / item['file_name']))
        if image is None: return self.__getitem__((idx + 1) % len(self.data_list))
        h_orig, w_orig = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = np.zeros((640, 640), dtype=np.float32)
        for word in item['words'].values():
            pts = np.array(word['points'], dtype=np.float32)
            pts[:, 0] *= 640 / w_orig
            pts[:, 1] *= 640 / h_orig
            cv2.fillPoly(gt_mask, [pts.astype(np.int32)], 1.0)
        if self.transform: image = self.transform(image)
        return {'image': image, 'gt_maps': torch.from_numpy(gt_mask).unsqueeze(0)}

def train():
    os.makedirs("checkpoints", exist_ok=True)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((640, 640)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    loader = DataLoader(OCRDataset(r"data/receipts/train/images", r"data/receipts/train/labels", transform), batch_size=2, shuffle=True)
    model = DBNetModule()
    trainer = L.Trainer(max_epochs=20)
    trainer.fit(model, loader)
    save_path = r"C:\Users\user\Documents\receipt_recommender_project\checkpoints\final_model_v2.ckpt"
    trainer.save_checkpoint(save_path)
    print(f"✅ 모델 생성 완료: {save_path}")

if __name__ == "__main__": train()