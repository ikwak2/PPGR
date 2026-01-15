# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from tqdm.auto import tqdm
import random

# [시각화 관련 라이브러리]
import matplotlib
matplotlib.use('Agg') # 서버 환경에서 화면 없이 그리기 필수 설정
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ==========================================
# 1. 환경 설정 & 하이퍼파라미터
# ==========================================
# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

NUM_USERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [SE-ResNet2 Spec]
D_MODEL = 256
NUM_HEADS = 8
DROPOUT_RATE = 0.4
SE_REDUCTION = 16  # SE Layer의 Reduction Ratio (보통 16 사용)

BATCH_SIZE = 128
LR = 0.001
EPOCHS = 30

LAMBDA_A = 0.5
LAMBDA_S = 0.01

print(f"🚀 [Start] SE-ResNet2 + Concat Fusion + Visualization Ready", flush=True)
print(f"⚙️  Device: {DEVICE} | Epochs: {EPOCHS} | SE Reduction: {SE_REDUCTION}", flush=True)

# ==========================================
# 2. 데이터셋 클래스 (변경 없음)
# ==========================================
class TriModalDataset(Dataset):
    def __init__(self, data_folder, window_size=300, stride=50, fs=128, mode='train', split_ratio=0.9):
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.mode = mode
        self.split_ratio = split_ratio
        
        self.samples_ppg = []
        self.samples_temp = []
        self.samples_acc = []
        self.labels = []
        
        search_pattern = os.path.join(data_folder, "user_*.csv")
        file_list = glob.glob(search_pattern)
        
        if not file_list:
            print(f"❌ Error: 데이터 파일을 찾을 수 없습니다. 경로: {data_folder}")
            return

        print(f"📂 [{mode.upper()}] 데이터 로딩 중... (파일 {len(file_list)}개)")
        
        for filepath in file_list:
            try:
                filename = os.path.basename(filepath)
                try:
                    user_num = int(filename.split('_')[1].split('.')[0])
                except:
                    user_num = int(filename.split('_')[1])

                df = pd.read_csv(filepath)
                df.columns = [c.strip() for c in df.columns]

                # 특정 사용자 데이터 분할 처리
                if user_num == 4:
                    df_segments = [df.iloc[:3786928], df.iloc[4194811:]]
                elif user_num == 6:
                    df_segments = [df.iloc[:4337569], df.iloc[4545544:]]
                else:
                    df_segments = [df]
                
                label = user_num - 1
                required_cols = ['PPG', 'temperature', 'acc_x', 'acc_y', 'acc_z']
                
                user_ppg, user_temp, user_acc, user_lbl = [], [], [], []

                for segment_df in df_segments:
                    if segment_df.empty: continue
                    if not all(col in segment_df.columns for col in required_cols): continue

                    raw_ppg = segment_df['PPG'].values
                    raw_temp = segment_df['temperature'].values
                    raw_acc = segment_df[['acc_x', 'acc_y', 'acc_z']].values.T

                    # 전처리
                    detrended = signal.detrend(raw_ppg)
                    b, a = signal.butter(4, [0.5/(0.5*fs), 8.0/(0.5*fs)], btype='band')
                    filtered = signal.filtfilt(b, a, detrended)
                    processed_ppg = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
                    
                    processed_temp = (raw_temp - 25.0) / (40.0 - 25.0)

                    acc_mean = np.mean(raw_acc, axis=1, keepdims=True)
                    acc_std = np.std(raw_acc, axis=1, keepdims=True) + 1e-6
                    processed_acc = (raw_acc - acc_mean) / acc_std

                    num_windows = (len(processed_ppg) - window_size) // stride
                    if num_windows <= 0: continue

                    for i in range(num_windows):
                        start = i * stride
                        end = start + window_size
                        user_ppg.append(processed_ppg[start:end])
                        user_temp.append(processed_temp[start:end])
                        user_acc.append(processed_acc[:, start:end])
                        user_lbl.append(label)

                total_len = len(user_ppg)
                split_idx = int(total_len * self.split_ratio)
                
                if self.mode == 'train':
                    self.samples_ppg.extend(user_ppg[:split_idx])
                    self.samples_temp.extend(user_temp[:split_idx])
                    self.samples_acc.extend(user_acc[:split_idx])
                    self.labels.extend(user_lbl[:split_idx])
                else:
                    self.samples_ppg.extend(user_ppg[split_idx:])
                    self.samples_temp.extend(user_temp[split_idx:])
                    self.samples_acc.extend(user_acc[split_idx:])
                    self.labels.extend(user_lbl[split_idx:])
                        
            except Exception as e:
                print(f"❌ 로드 에러 {filename}: {e}")

        self.samples_ppg = np.array(self.samples_ppg, dtype=np.float32)
        self.samples_temp = np.array(self.samples_temp, dtype=np.float32)
        self.samples_acc = np.array(self.samples_acc, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        if len(self.labels) > 0:
            self.samples_ppg = np.expand_dims(self.samples_ppg, axis=1)
            self.samples_temp = np.expand_dims(self.samples_temp, axis=1)

        print(f"🎉 [{mode.upper()}] 로드 완료! 샘플 수: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples_ppg[idx]),
                torch.from_numpy(self.samples_temp[idx]),
                torch.from_numpy(self.samples_acc[idx])), torch.tensor(self.labels[idx])

# ==========================================
# 3. 모델 아키텍처: SE-ResNet2 + Concat Fusion
# ==========================================

# [NEW] SE Layer 정의
class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        # Squeeze: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # Scale: 원본 특징 맵에 중요도(y)를 곱함
        return x * y

# [NEW] SEBasicBlock 정의 (기존 BasicBlock 대체)
class SEBasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEBasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE Layer 추가
        self.se = SELayer1D(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Residual 더하기 전에 SE 적용 (Recalibration)
        out = self.se(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet2Encoder(nn.Module):
    def __init__(self, in_channels=1, d_model=256):
        super(ResNet2Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # BasicBlock 대신 SEBasicBlock 사용
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        # Grad-CAM 타겟용
        self.layer3 = self._make_layer(128, d_model, blocks=6, stride=2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = [SEBasicBlock1D(in_planes, out_planes, stride, reduction=SE_REDUCTION)]
        for _ in range(1, blocks):
            layers.append(SEBasicBlock1D(out_planes, out_planes, reduction=SE_REDUCTION))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # (B, d_model, Seq_len) -> Global Average Pooling -> (B, d_model)
        return self.pool(x).squeeze(-1), x.transpose(1, 2)

class TriModalFusion(nn.Module):
    def __init__(self, num_users=16, d_model=256, num_heads=8):
        super(TriModalFusion, self).__init__()
        # Encoders
        self.ppg_encoder = ResNet2Encoder(in_channels=1, d_model=d_model)
        self.temp_encoder = ResNet2Encoder(in_channels=1, d_model=d_model)
        self.acc_encoder = ResNet2Encoder(in_channels=3, d_model=d_model)
        
        # Attention
        self.cross_att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        
        # Fusion (Concat + Projection)
        self.fusion_projector = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, num_users)
        )

    def forward(self, x_ppg, x_temp, x_acc):
        z_p, seq_p = self.ppg_encoder(x_ppg)
        z_t, seq_t = self.temp_encoder(x_temp)
        z_a, seq_a = self.acc_encoder(x_acc)
        
        att_p, _ = self.cross_att(z_p.unsqueeze(1), seq_t, seq_t)
        z_p_r = self.norm_p(z_p + att_p.squeeze(1))
        
        att_t, _ = self.cross_att(z_t.unsqueeze(1), seq_p, seq_p)
        z_t_r = self.norm_t(z_t + att_t.squeeze(1))
        
        z_a_r = self.norm_a(z_a)

        # Concat Fusion
        combined = torch.cat([z_p_r, z_t_r, z_a_r], dim=1)
        z_fused = self.fusion_projector(combined)
        z_fused_final = F.normalize(z_fused, p=2, dim=1)
        
        return self.classifier(z_fused_final), z_fused_final, z_p_r, z_t_r, z_a_r

# ==========================================
# 4. 손실 함수 (변경 없음)
# ==========================================
class AlignmentLoss(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z_a, z_b):
        logits = torch.matmul(F.normalize(z_a, dim=1), F.normalize(z_b, dim=1).T) / self.t
        labels = torch.arange(z_a.size(0)).to(z_a.device)
        return self.ce(logits, labels)

class SpreadControlLoss(nn.Module):
    def __init__(self, th=0.001):
        super().__init__()
        self.th = th
    def forward(self, z):
        return F.relu(torch.var(F.normalize(z, dim=1), dim=0).mean() - self.th)

# ==========================================
# 5. 평가 및 시각화 도구 함수들 (변경 없음)
# ==========================================

def plot_training_history(history, save_path='training_curves_se.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Loss Curves (SE-ResNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy (SE-ResNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 학습 곡선 저장 완료: {save_path}")

def calculate_eer(genuine_scores, impostor_scores):
    scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(scores.min() - 0.01, scores.max() + 0.01, 1000)
    far = np.array([np.sum(impostor_scores >= t) / len(impostor_scores) for t in thresholds])
    frr = np.array([np.sum(genuine_scores < t) / len(genuine_scores) for t in thresholds])
    diff = np.abs(far - frr)
    eer_idx = np.argmin(diff)
    return (far[eer_idx] + frr[eer_idx]) / 2

def get_embeddings_and_scores(model, data_loader, device, num_users):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for (ppg, temp, acc), labels in tqdm(data_loader, desc="[Extracting]"):
            ppg, temp, acc = ppg.to(device), temp.to(device), acc.to(device)
            _, z_fused_final, _, _, _ = model(ppg, temp, acc)
            all_embeddings.append(z_fused_final.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # EER 계산을 위한 점수 생성
    user_templates = {}
    for user_id in range(num_users):
        user_embs = all_embeddings[all_labels == user_id]
        if len(user_embs) > 0:
            user_templates[user_id] = np.mean(user_embs, axis=0)
        else:
            user_templates[user_id] = None
            
    genuine_scores, impostor_scores = [], []
    for emb, label in zip(all_embeddings, all_labels):
        target_template = user_templates.get(label)
        if target_template is None: continue
        sim_g = F.cosine_similarity(torch.from_numpy(emb).unsqueeze(0), torch.from_numpy(target_template).unsqueeze(0)).item()
        genuine_scores.append(sim_g)
        for other_id, other_template in user_templates.items():
            if other_id != label and other_template is not None:
                sim_i = F.cosine_similarity(torch.from_numpy(emb).unsqueeze(0), torch.from_numpy(other_template).unsqueeze(0)).item()
                impostor_scores.append(sim_i)
                
    return all_embeddings, all_labels, np.array(genuine_scores), np.array(impostor_scores)

def visualize_tsne(embeddings, labels, save_path='tsne_result_se.png', max_samples=2000):
    print("🎨 t-SNE 계산 및 시각화 중...", flush=True)
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    df_tsne = pd.DataFrame({'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 'User': labels})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='User', palette='tab20', s=60, alpha=0.7)
    plt.title('t-SNE (SE-ResNet Embeddings)', fontsize=15)
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.legend(title='User ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ t-SNE 저장 완료: {save_path}")

class GradCAM1D:
    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.handles = []
        for name, layer in target_layers.items():
            self.handles.append(layer.register_forward_hook(self.save_activation(name)))
            self.handles.append(layer.register_backward_hook(self.save_gradient(name)))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook

    def __call__(self, x_ppg, x_temp, x_acc, target_class_idx):
        self.model.eval()
        self.model.zero_grad()
        logits, _, _, _, _ = self.model(x_ppg, x_temp, x_acc)
        if target_class_idx is None:
            target_class_idx = logits.argmax(dim=1).item()
        target_score = logits[:, target_class_idx]
        target_score.backward(retain_graph=True)
        cams = {}
        for name, grads in self.gradients.items():
            acts = self.activations[name]
            weights = torch.mean(grads, dim=2, keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=300, mode='linear', align_corners=False)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
            cams[name] = cam.squeeze().cpu().numpy()
        return cams

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

def visualize_gradcam_samples(model, data_loader, device, save_path='gradcam_analysis_se.png', num_samples=3):
    print("🧠 Grad-CAM 분석 및 시각화 중...", flush=True)
    model.eval()
    target_layers = {
        'PPG': model.ppg_encoder.layer3[-1],
        'Temp': model.temp_encoder.layer3[-1],
        'Acc': model.acc_encoder.layer3[-1]
    }
    gradcam = GradCAM1D(model, target_layers)
    data_iter = iter(data_loader)
    ppg_batch, temp_batch, acc_batch, label_batch = next(data_iter)
    
    plt.figure(figsize=(15, 4 * num_samples))
    for i in range(num_samples):
        ppg = ppg_batch[i].unsqueeze(0).to(device)
        temp = temp_batch[i].unsqueeze(0).to(device)
        acc = acc_batch[i].unsqueeze(0).to(device)
        label = label_batch[i].item()
        cams = gradcam(ppg, temp, acc, target_class_idx=label)
        x_axis = np.arange(300)
        
        plt.subplot(num_samples, 3, i*3 + 1)
        ppg_signal = ppg.squeeze().cpu().numpy()
        plt.plot(x_axis, ppg_signal, 'b-', alpha=0.6, label='PPG Signal')
        plt.imshow(cams['PPG'].reshape(1, -1), cmap='jet', aspect='auto', extent=[0, 300, ppg_signal.min(), ppg_signal.max()], alpha=0.5)
        plt.title(f'User {label}: PPG Importance')
        if i==0: plt.legend()

        plt.subplot(num_samples, 3, i*3 + 2)
        temp_signal = temp.squeeze().cpu().numpy()
        plt.plot(x_axis, temp_signal, 'g-', alpha=0.6, label='Temp Signal')
        plt.imshow(cams['Temp'].reshape(1, -1), cmap='jet', aspect='auto', extent=[0, 300, temp_signal.min()-0.1, temp_signal.max()+0.1], alpha=0.5)
        plt.title(f'User {label}: Temp Importance')
        if i==0: plt.legend()

        plt.subplot(num_samples, 3, i*3 + 3)
        acc_signal_x = acc.squeeze().cpu().numpy()[0]
        plt.plot(x_axis, acc_signal_x, 'k-', alpha=0.6, label='Acc(X) Signal')
        plt.imshow(cams['Acc'].reshape(1, -1), cmap='jet', aspect='auto', extent=[0, 300, acc_signal_x.min(), acc_signal_x.max()], alpha=0.5)
        plt.title(f'User {label}: Acc(X) Importance')
        if i==0: plt.legend()
        
    gradcam.remove_hooks()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Grad-CAM 저장 완료: {save_path}")

# ==========================================
# 6. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    data_folder = "/Data/CRS25/PPG_Certifiation/data/Final_Data"
    
    train_dataset = TriModalDataset(data_folder, mode='train', split_ratio=0.9)
    val_dataset = TriModalDataset(data_folder, mode='val', split_ratio=0.9)
    
    if len(train_dataset) == 0: exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = TriModalFusion(num_users=NUM_USERS, d_model=D_MODEL, num_heads=NUM_HEADS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    crit_cls = nn.CrossEntropyLoss()
    crit_align = AlignmentLoss()
    crit_spread = SpreadControlLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\n🔥 학습 시작 (총 {EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for (ppg, temp, acc), labels in pbar:
            ppg, temp, acc, labels = ppg.to(DEVICE), temp.to(DEVICE), acc.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out, z_final, z_p, z_t, z_a = model(ppg, temp, acc)
            
            loss_cls = crit_cls(out, labels)
            loss_align = crit_align(z_p, z_t) + crit_align(z_t, z_p)
            loss_spread = crit_spread(z_p) + crit_spread(z_t) + crit_spread(z_a)
            total_loss = loss_cls + (LAMBDA_A * loss_align) + (LAMBDA_S * loss_spread)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}", 'Acc': f"{100*correct/total:.1f}%"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for (ppg, temp, acc), labels in val_loader:
                ppg, temp, acc, labels = ppg.to(DEVICE), temp.to(DEVICE), acc.to(DEVICE), labels.to(DEVICE)
                out, _, _, _, _ = model(ppg, temp, acc)
                loss = crit_cls(out, labels)
                val_loss += loss.item()
                _, pred = torch.max(out.data, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"✨ [Ep {epoch+1}] Tr Loss: {avg_train_loss:.4f} (Acc {train_acc:.1f}%) | Val Loss: {avg_val_loss:.4f} (Acc {val_acc:.1f}%)")

    print("\n🏁 학습 종료. 최종 평가 및 시각화 진행 중...")
    
    plot_training_history(history)
    embeddings, labels, gen_scores, imp_scores = get_embeddings_and_scores(model, val_loader, DEVICE, NUM_USERS)
    if len(gen_scores) > 0:
        final_eer = calculate_eer(gen_scores, imp_scores)
        print(f"\n🏆 최종 결과 (SE-ResNet2 + Concat)")
        print(f"   - Validation Accuracy : {val_acc:.2f}%")
        print(f"   - EER (Equal Error Rate): {final_eer * 100:.4f}%")
    
    visualize_tsne(embeddings, labels)
    
    viz_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    visualize_gradcam_samples(model, viz_loader, DEVICE, num_samples=3)

    print("\n🎉 모든 작업이 완료되었습니다. 생성된 이미지 파일들을 확인하세요.")