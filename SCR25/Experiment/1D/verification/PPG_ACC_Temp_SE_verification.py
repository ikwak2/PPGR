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
import math
import time 

# [시각화 및 평가 관련]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ==========================================
# 1. 환경 설정 & 하이퍼파라미터
# ==========================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# [수정 1] 16명 -> 12명으로 변경
NUM_USERS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습 설정
D_MODEL = 256
NUM_HEADS = 8
DROPOUT_RATE = 0.2
SE_REDUCTION = 16
WINDOW_SIZE = 128 * 4 # 4초로 설정 
STRIDE = WINDOW_SIZE // 2 # 50% Overlap
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 30

# [수정 4] Early Stopping 파라미터 추가
EARLY_STOPPING_PATIENCE = 5  

# AAM-Softmax (ArcFace) 파라미터
S = 30.0  
M = 0.50  

print(f"🚀 [Real-time Verification Experiment] Session-based Test & Enrollment")
print(f"⚙️  25s Stabilization Drop Applied | L2 Norm Fixed | Val EER Monitored | Users: {NUM_USERS}")

# ==========================================
# 2. 데이터셋 클래스
# ==========================================
class VerificationDataset(Dataset):
    def __init__(self, data_folder, window_size=300, stride=150, fs=128, mode='train', train_val_ratio=0.8):
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.mode = mode
        
        self.test_duration = 30 * 60 * fs      
        self.enroll_duration = 5 * 60 * fs     
        self.session_duration = 3 * 60 * fs    
        self.drop_samples = 25 * fs            
        self.windows_per_session = 10          
        
        self.samples_ppg = []
        self.samples_temp = []
        self.samples_acc = []
        self.labels = []
        self.session_ids = [] 
        
        search_pattern = os.path.join(data_folder, "user_*.csv")
        file_list = sorted(glob.glob(search_pattern))
        
        for filepath in file_list:
            filename = os.path.basename(filepath)
            user_num = int(''.join(filter(str.isdigit, filename)))
            label = user_num - 1
            
            # [수정 1] 지정된 NUM_USERS(12명)까지만 데이터 로드
            if label >= NUM_USERS:
                continue
            
            df = pd.read_csv(filepath)
            df.columns = [c.strip() for c in df.columns]

            raw_ppg = df['PPG'].values
            detrended = signal.detrend(raw_ppg)
            b, a = signal.butter(4, [0.5/(0.5*fs), 8.0/(0.5*fs)], btype='band')
            filtered = signal.filtfilt(b, a, detrended)
            processed_ppg = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
            processed_temp = (df['temperature'].values - 25.0) / (40.0 - 25.0)
            raw_acc = df[['acc_x', 'acc_y', 'acc_z']].values.T
            processed_acc = (raw_acc - np.mean(raw_acc, axis=1, keepdims=True)) / (np.std(raw_acc, axis=1, keepdims=True) + 1e-6)

            total_samples = len(processed_ppg)
            test_start_idx = total_samples - self.test_duration
            enroll_start_idx = test_start_idx - self.enroll_duration
            
            if self.mode in ['train', 'val']:
                pool_ppg = processed_ppg[:enroll_start_idx]
                pool_temp = processed_temp[:enroll_start_idx]
                pool_acc = processed_acc[:, :enroll_start_idx]
                
                split_idx = int(len(pool_ppg) * train_val_ratio)
                if self.mode == 'train':
                    u_ppg, u_temp, u_acc = pool_ppg[:split_idx], pool_temp[:split_idx], pool_acc[:, :split_idx]
                else: 
                    u_ppg, u_temp, u_acc = pool_ppg[split_idx:], pool_temp[split_idx:], pool_acc[:, split_idx:]

                num_windows = (len(u_ppg) - self.window_size) // self.stride
                for i in range(num_windows):
                    start = i * self.stride
                    self._add_sample(u_ppg, u_temp, u_acc, start, label, session_id=0)

            elif self.mode == 'enroll':
                u_ppg = processed_ppg[enroll_start_idx:test_start_idx]
                u_temp = processed_temp[enroll_start_idx:test_start_idx]
                u_acc = processed_acc[:, enroll_start_idx:test_start_idx]
                
                base_start = self.drop_samples
                valid_length = len(u_ppg) - base_start
                num_candidates = valid_length // self.window_size
                
                candidates = []
                for i in range(num_candidates):
                    start = base_start + (i * self.window_size)
                    end = start + self.window_size
                    acc_window = u_acc[:, start:end]
                    motion_noise = np.sum(np.var(acc_window, axis=1))
                    candidates.append((start, motion_noise))
                
                candidates.sort(key=lambda x: x[1])
                best_starts = [cand[0] for cand in candidates[:self.windows_per_session]]
                best_starts.sort()
                
                for start in best_starts:
                    self._add_sample(u_ppg, u_temp, u_acc, start, label, session_id=0)

            elif self.mode == 'test':
                u_ppg = processed_ppg[test_start_idx:]
                u_temp = processed_temp[test_start_idx:]
                u_acc = processed_acc[:, test_start_idx:]
                
                for s in range(10):
                    sess_start_idx = s * self.session_duration
                    base_start = sess_start_idx + self.drop_samples
                    for i in range(self.windows_per_session):
                        start = base_start + (i * self.window_size)
                        self._add_sample(u_ppg, u_temp, u_acc, start, label, session_id=s)

        self.samples_ppg = np.expand_dims(np.array(self.samples_ppg, dtype=np.float32), 1)
        self.samples_temp = np.expand_dims(np.array(self.samples_temp, dtype=np.float32), 1)
        self.samples_acc = np.array(self.samples_acc, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.session_ids = np.array(self.session_ids, dtype=np.int64)

    def _add_sample(self, ppg, temp, acc, start, label, session_id):
        end = start + self.window_size
        self.samples_ppg.append(ppg[start:end])
        self.samples_temp.append(temp[start:end])
        self.samples_acc.append(acc[:, start:end])
        self.labels.append(label)
        self.session_ids.append(session_id)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples_ppg[idx]),
                torch.from_numpy(self.samples_temp[idx]),
                torch.from_numpy(self.samples_acc[idx])), torch.tensor(self.labels[idx]), self.session_ids[idx]

# ==========================================
# 3. 모델 컴포넌트
# ==========================================
class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class SEBasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SELayer1D(out_channels, reduction)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out))) + self.shortcut(x)
        return self.relu(out)

class ResNet2Encoder(nn.Module):
    def __init__(self, in_channels=1, d_model=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3, 1)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, d_model, 6, 2)
        self.pool = nn.AdaptiveAvgPool1d(1) 
    def _make_layer(self, in_p, out_p, blocks, stride):
        layers = [SEBasicBlock1D(in_p, out_p, stride, reduction=SE_REDUCTION)]
        for _ in range(1, blocks): layers.append(SEBasicBlock1D(out_p, out_p, reduction=SE_REDUCTION))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.pool(x).squeeze(-1), x.transpose(1, 2)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = torch.where(cosine > self.th, cosine * self.cos_m - sine * self.sin_m, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device).scatter_(1, label.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

class VerificationModel(nn.Module):
    def __init__(self, num_users=16, d_model=256):
        super().__init__()
        self.ppg_enc, self.tmp_enc, self.acc_enc = ResNet2Encoder(1, d_model), ResNet2Encoder(1, d_model), ResNet2Encoder(3, d_model)
        self.fusion_projector = nn.Sequential(nn.Linear(d_model * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, d_model))
        self.arcface = ArcMarginProduct(d_model, num_users, s=S, m=M)
    def forward(self, x_p, x_t, x_a, labels=None):
        z_p, _ = self.ppg_enc(x_p)
        z_t, _ = self.tmp_enc(x_t)
        z_a, _ = self.acc_enc(x_a)
        z_norm = F.normalize(self.fusion_projector(torch.cat([z_p, z_t, z_a], dim=1)), p=2, dim=1)
        if labels is not None: return self.arcface(z_norm, labels), z_norm
        return None, z_norm

# ==========================================
# 4. 평가 함수들
# ==========================================
def calculate_eer(genuine_scores, impostor_scores):
    scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    far = np.array([np.sum(impostor_scores >= t) / len(impostor_scores) for t in thresholds])
    frr = np.array([np.sum(genuine_scores < t) / len(genuine_scores) for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    return (far[idx] + frr[idx]) / 2, thresholds[idx]

def extract_embeddings(model, loader):
    model.eval()
    all_embs, all_lbls, all_sess = [], [], []
    with torch.no_grad():
        for (p, t, a), lbl, sess in loader:
            _, feat = model(p.to(DEVICE), t.to(DEVICE), a.to(DEVICE))
            all_embs.append(feat.cpu().numpy())
            all_lbls.append(lbl.numpy())
            all_sess.append(sess.numpy())
    return np.concatenate(all_embs), np.concatenate(all_lbls), np.concatenate(all_sess)

def evaluate_val_eer(model, val_loader, num_users):
    embs, lbls, _ = extract_embeddings(model, val_loader)
    templates, gen_scores, imp_scores = {}, [], []
    
    for u in range(num_users):
        u_idx = np.where(lbls == u)[0]
        if len(u_idx) < 5: continue
        raw_template = np.mean(embs[u_idx[:5]], axis=0)
        templates[u] = raw_template / (np.linalg.norm(raw_template) + 1e-8)
        
        for i in u_idx[5:]:
            gen_scores.append(np.dot(embs[i], templates[u]))
            for other_u in range(num_users):
                if other_u != u and other_u in templates:
                    imp_scores.append(np.dot(embs[i], templates[other_u]))
                    
    eer, _ = calculate_eer(np.array(gen_scores), np.array(imp_scores))
    return eer

def evaluate_realtime_verification(model, enroll_loader, test_loader, num_users):
    en_embs, en_lbls, _ = extract_embeddings(model, enroll_loader)
    templates = {}
    for u in range(num_users):
        u_idx = np.where(en_lbls == u)[0]
        raw_template = np.mean(en_embs[u_idx], axis=0)
        templates[u] = raw_template / (np.linalg.norm(raw_template) + 1e-8)
        
    ts_embs, ts_lbls, ts_sess = extract_embeddings(model, test_loader)
    session_eers = []
    
    for s in range(10): 
        s_idx = np.where(ts_sess == s)[0]
        curr_embs, curr_lbls = ts_embs[s_idx], ts_lbls[s_idx]
        
        gen_scores, imp_scores = [], []
        for i, emb in enumerate(curr_embs):
            true_u = curr_lbls[i]
            for u, template in templates.items():
                sim = np.dot(emb, template)
                if u == true_u: gen_scores.append(sim)
                else: imp_scores.append(sim)
                
        eer, _ = calculate_eer(np.array(gen_scores), np.array(imp_scores))
        session_eers.append(eer)
        
    return np.mean(session_eers), session_eers

# ==========================================
# 5. 메인 실행 루틴
# ==========================================
if __name__ == "__main__":
    data_path = "/Data/CRS25/PPG_Certifiation/data/Final_Data"
    
    train_ds = VerificationDataset(data_path, mode='train')
    val_ds = VerificationDataset(data_path, mode='val')
    enroll_ds = VerificationDataset(data_path, mode='enroll')
    test_ds = VerificationDataset(data_path, mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    enroll_loader = DataLoader(enroll_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = VerificationModel(NUM_USERS, D_MODEL).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # [수정 4 & 5] Early Stopping 관련 변수 초기화
    best_val_eer = float('inf')
    patience_counter = 0
    best_model_path = "best_biometric_model.pt"

    print("\n🔥 Starting Training with ArcFace...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        
        for (p, t, a), lbl, _ in tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}"):
            p, t, a, lbl = p.to(DEVICE), t.to(DEVICE), a.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(p, t, a, lbl)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)

        # [수정 2] Train Accuracy 계산
        train_acc = 100 * correct / total

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for (p, t, a), lbl, _ in val_loader:
                p, t, a, lbl = p.to(DEVICE), t.to(DEVICE), a.to(DEVICE), lbl.to(DEVICE)
                _, feats = model(p, t, a, labels=None) 
                
                weights_norm = F.normalize(model.arcface.weight, p=2, dim=1)
                logits = torch.mm(feats, weights_norm.t()) * S
                _, pred = torch.max(logits, 1)
                v_correct += (pred == lbl).sum().item()
                v_total += lbl.size(0)
        
        v_acc = 100 * v_correct / v_total
        val_eer = evaluate_val_eer(model, val_loader, NUM_USERS) * 100 
        
        # [수정 2] 출력문에 Train Acc 추가
        print(f"   Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {v_acc:.2f}% | Val EER: {val_eer:.2f}%")

        # [수정 4 & 5] Early Stopping 로직 및 최적 모델 저장
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   [Model Saved] Best Val EER updated to {best_val_eer:.2f}%!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}! Best Val EER was {best_val_eer:.2f}%")
                break

    # --- 최종 평가 (Real-time Session Simulation) ---
    print("\n🏁 Final Evaluation on 10 Independent Sessions (30-min Test Set)...")
    
    # [수정 5] 평가 전, 저장해둔 가장 성능이 좋았던 모델 가중치 로드
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    
    # [수정 3] Final Evaluation 소요 시간 측정 시작
    eval_start_time = time.time()
    avg_eer, session_eers = evaluate_realtime_verification(model, enroll_loader, test_loader, NUM_USERS)
    eval_duration = time.time() - eval_start_time
    
    print("\n📊 [Session-by-Session EER Results]")
    for s, s_eer in enumerate(session_eers):
        print(f"   Session {s+1} (3min): {s_eer*100:.3f}%")
        
    print(f"\n🏆 Final Average EER (Robustness): {avg_eer*100:.4f}%")
    # [수정 3] 소요 시간 출력
    print(f"⏱️ Evaluation Time Taken: {eval_duration:.2f} seconds")