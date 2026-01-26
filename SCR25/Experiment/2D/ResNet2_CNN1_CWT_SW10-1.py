# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 화면 출력 없이 파일 저장 모드
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms


# ==========================================
# 1. 설정 (Configuration)
# ==========================================
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DATA_DIR = "./processed_data_10sec" # 전처리된 데이터 경로
NUM_USERS = 16
BATCH_SIZE = 8
EPOCHS = 30
LR = 0.0001 # Pre-trained 모델이라 학습률을 조금 낮게 시작
DEVICE = torch.device("cuda:1")
torch.cuda.set_device(1)


print(f"⚙️ Device: {DEVICE}")

# ==========================================
# 2. 데이터셋 클래스 (HybridDataset)
# ==========================================
class HybridDataset(Dataset):
    def __init__(self, data_dir, mode='train', split_ratio=0.8):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.sig_dir = os.path.join(data_dir, "signals")
        
        meta_path = os.path.join(data_dir, "metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"❌ 메타데이터가 없습니다: {meta_path}")
            
        full_df = pd.read_csv(meta_path)
        
        train_indices = []
        val_indices = []
        
        # 사용자별로 루프를 돌며 시간 순서대로 8:2 분할
        for label in sorted(full_df['label'].unique()):
            # 1. 해당 사용자의 데이터만 추출
            user_df = full_df[full_df['label'] == label]
            
            # 2. 시간 순서 보장을 위해 sample_id 기준 정렬
            user_df = user_df.sort_values(by='sample_id')
            
            # 3. 분할 지점 계산 (비율 기반)
            n_samples = len(user_df)
            split_point = int(n_samples * split_ratio)
            
            # 4. 전체 DataFrame 기준의 인덱스 가져오기
            indices = user_df.index.tolist()
            
            # 5. 순서대로 앞쪽은 Train, 뒤쪽은 Val
            train_indices.extend(indices[:split_point])
            val_indices.extend(indices[split_point:])
            
        # 모드에 따라 데이터프레임 재구성
        if mode == 'train':
            self.data = full_df.loc[train_indices].reset_index(drop=True)
        else:
            self.data = full_df.loc[val_indices].reset_index(drop=True)
            
        # 디버깅용 출력
        print(f"[{mode.upper()}] Dataset Created. Total Samples: {len(self.data)}")
        if len(self.data) > 0:
            print(f"   - Users included: {sorted(self.data['label'].unique())}")
            print(f"   - Sample range: {self.data.iloc[0]['sample_id']} ~ {self.data.iloc[-1]['sample_id']}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['sample_id']
        label = row['label']
        
        # 이미지 로드
        img_path = os.path.join(self.img_dir, f"{sample_id}.png")
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # 신호 로드
        sig_path = os.path.join(self.sig_dir, f"{sample_id}.npy")
        # mmap_mode='r'은 대용량 파일 읽을 때 유용하나, 작은 파일 수만 개를 읽을 땐 오버헤드가 될 수 있어 제거해도 무방합니다.
        # 여기서는 안정성을 위해 일반 로드로 변경하거나 유지하셔도 됩니다.
        signal = np.load(sig_path) 
        signal = signal.T.astype(np.float32) # (Length, Channel) -> (Channel, Length)
        signal_tensor = torch.from_numpy(signal)
        
        return image_tensor, signal_tensor, torch.tensor(label, dtype=torch.long)
    


# ==========================================
# 3. 모델 아키텍처 (Hybrid Fusion Model)
# ==========================================
class Sensor1DCNN(nn.Module):
    """Acc/Temp 처리를 위한 1D CNN"""
    def __init__(self, in_channels=4, out_dim=128):
        super(Sensor1DCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1) # (B, 128, 1) -> (B, 128)
        )
        self.fc = nn.Linear(128, out_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.fc(x)

class HybridFusionModel(nn.Module):
    def __init__(self, num_classes=16):
        super(HybridFusionModel, self).__init__()
        
        # --- Branch 1: PPG (2D CNN - ResNet18) ---
        # Pre-trained ResNet18 로드
        self.ppg_backbone = models.resnet18(pretrained=True)
        # 마지막 FC 레이어 교체 (분류 대신 특징 추출용)
        # ResNet18의 fc input feature는 512
        self.ppg_feature_dim = 512
        self.ppg_backbone.fc = nn.Linear(self.ppg_backbone.fc.in_features, self.ppg_feature_dim)
        
        # --- Branch 2: Sensor (1D CNN) ---
        self.sensor_feature_dim = 128
        self.sensor_net = Sensor1DCNN(in_channels=4, out_dim=self.sensor_feature_dim)
        
        # --- Fusion & Classifier ---
        combined_dim = self.ppg_feature_dim + self.sensor_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, img, sensor):
        # Extract features
        ppg_feat = self.ppg_backbone(img)       # (B, 512)
        sensor_feat = self.sensor_net(sensor)   # (B, 128)
        
        # Concatenate
        combined = torch.cat((ppg_feat, sensor_feat), dim=1) # (B, 640)
        
        # Classify
        logits = self.classifier(combined)
        
        # 임베딩 시각화를 위해 특징 벡터(combined)도 반환
        return logits, combined

# ==========================================
# 4. 학습 및 시각화 유틸리티
# ==========================================

def safe_collate(batch):
    imgs, sigs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    sigs = torch.stack(sigs, dim=0)
    labels = torch.tensor(labels)
    return imgs, sigs, labels


def train_model():
    # 데이터셋 로드 (8:2 비율 적용)
    train_dataset = HybridDataset(
        data_dir=DATA_DIR, 
        mode='train', 
        split_ratio=0.8
    )
    
    val_dataset = HybridDataset(
        data_dir=DATA_DIR, 
        mode='val', 
        split_ratio=0.8
    )
    
    # DataLoader 설정 (이전 대화에서 최적화한 설정 유지: GPU 1개 사용, num_workers=0)
    # RTX 8000 단독 사용 시 배치 사이즈 512 추천
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, # 512 권장
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        collate_fn=safe_collate
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📊 Data Loaded: Train {len(train_dataset)}, Val {len(val_dataset)}")
    
    # 데이터 범위 확인 로그
    for u in range(16):
        user_train = train_dataset.data[train_dataset.data['label'] == u]
        user_val = val_dataset.data[val_dataset.data['label'] == u]
        
        if not user_train.empty and not user_val.empty:
            print(f"User {u}: Train[{len(user_train)}] Val[{len(user_val)}]")

    # 모델 초기화
    model = HybridFusionModel(num_classes=NUM_USERS).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    # ResNet 파라미터는 학습률을 조금 낮게, 나머지는 높게 설정하는 기법도 있지만 여기선 통일
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("\n🔥 Training Start...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for imgs, sigs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs, sigs, labels = imgs.to(DEVICE), sigs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(imgs, sigs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, sigs, labels in val_loader:
                imgs, sigs, labels = imgs.to(DEVICE), sigs.to(DEVICE), labels.to(DEVICE)
                outputs, _ = model(imgs, sigs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_acc = 100 * correct / total
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_acc)
        
        print(f"✨ [Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
        scheduler.step(epoch_val_loss)
        
    return model, history, val_loader

def plot_curves(history, save_path='training_result.png'):
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 결과 그래프 저장됨: {save_path}")

def visualize_tsne(model, dataloader, device, save_path='tsne_plot.png'):
    print("🎨 t-SNE 시각화 생성 중...")
    model.eval()
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for imgs, sigs, labels in tqdm(dataloader, desc="Extracting Features"):
            imgs, sigs = imgs.to(device), sigs.to(device)
            _, feats = model(imgs, sigs) # (B, 640)
            
            embeddings.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    # 데이터가 너무 많으면 랜덤 샘플링 (속도 문제 시 주석 해제)
    if len(embeddings) > 3000:
       idx = np.random.choice(len(embeddings), 3000, replace=False)
       embeddings = embeddings[idx]
       labels_list = labels_list[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels_list, palette="tab20", s=60, alpha=0.7, legend='full')
    plt.title("t-SNE of Hybrid Model Embeddings")
    plt.legend(title='User ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ t-SNE 저장됨: {save_path}")

# ==========================================
# 5. 메인 실행
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        print("먼저 전처리 코드(preprocess_cwt.py)를 실행해주세요.")
    else:
        # 학습 실행
        trained_model, history, val_loader = train_model()
        
        # 그래프 그리기
        plot_curves(history)
        
        # t-SNE 시각화
        visualize_tsne(trained_model, val_loader, DEVICE)
        
        # 모델 저장
        torch.save(trained_model.state_dict(), "ResNet2_CNN1_CWT_SW10-1.pth")
        print("\n🎉 모든 작업 완료!")