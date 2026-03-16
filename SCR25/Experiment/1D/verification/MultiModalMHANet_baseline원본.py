import os
import glob
import math
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import roc_curve, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. Dataset & Preprocessing (실험 설계 완벽 반영)
# ==============================================================================
class BiometricDataSplitter:
    def __init__(self, data_folder, window_size=512, fs=128):
        self.data_folder = data_folder
        self.window_size = window_size # 4초 윈도우 (128 * 4)
        self.fs = fs
        self.train_stride = 256        # 2초 슬라이드 (128 * 2)
        
        self.train_data = {'ppg': [], 'temp': [], 'acc': [], 'label': []}
        self.val_data = {'ppg': [], 'temp': [], 'acc': [], 'label': []}
        self.enroll_data = {} 
        self.test_data = {}   

        self._process_all_users()

    def _process_all_users(self):
        file_list = sorted(glob.glob(os.path.join(self.data_folder, "user_*.csv")))
        print("📂 데이터 로드 및 실험 설계(Train/Val/Enroll/Test) 분할 진행 중...")
        
        for filepath in file_list:
            filename = os.path.basename(filepath)
            try: user_num = int(filename.split('_')[1].split('.')[0])
            except: user_num = int(filename.split('_')[1])
                
            if user_num > 12: continue # Closed-set 12명 기준

            df = pd.read_csv(filepath)
            df.columns = [c.strip() for c in df.columns]

            if user_num == 4: df_segments = [df.iloc[:3786928], df.iloc[4194811:]]
            elif user_num == 6: df_segments = [df.iloc[:4337569], df.iloc[4545544:]]
            else: df_segments = [df]

            label = user_num - 1
            
            raw_ppg, raw_temp, raw_acc = [], [], []
            for segment_df in df_segments:
                raw_ppg.extend(segment_df['PPG'].values)
                raw_temp.extend(segment_df['temperature'].values)
                raw_acc.append(segment_df[['acc_x', 'acc_y', 'acc_z']].values.T)
                
            raw_ppg = np.array(raw_ppg)
            raw_temp = np.array(raw_temp)
            raw_acc = np.concatenate(raw_acc, axis=1)

            detrended = signal.detrend(raw_ppg)
            b, a = signal.butter(4, [0.5/(0.5*self.fs), 8.0/(0.5*self.fs)], btype='band')
            filtered = signal.filtfilt(b, a, detrended)
            
            # --- [시간 분할 로직] 마지막 30분(Test) + 앞 5분(Enroll) + 나머지(Train/Val) ---
            total_len = len(filtered)
            test_len = 30 * 60 * self.fs     
            enroll_len = 5 * 60 * self.fs    
            past_len = total_len - test_len - enroll_len 
            
            if past_len <= 0: continue

            # 정규화 (Train 구간 통계량 활용 - Data Leakage 방지)
            train_ppg_mean = np.mean(filtered[:past_len])
            train_ppg_std = np.std(filtered[:past_len]) + 1e-6
            processed_ppg = (filtered - train_ppg_mean) / train_ppg_std
            
            processed_temp = (raw_temp - 25.0) / (40.0 - 25.0)
            
            train_acc_mean = np.mean(raw_acc[:, :past_len], axis=1, keepdims=True)
            train_acc_std = np.std(raw_acc[:, :past_len], axis=1, keepdims=True) + 1e-6
            processed_acc = (raw_acc - train_acc_mean) / train_acc_std

            # --- [1] Train / Val 분할 (8:2) ---
            train_len = int(past_len * 0.8)
            
            w_ppg, w_t, w_a = self._extract_windows(processed_ppg[:train_len], processed_temp[:train_len], processed_acc[:, :train_len], self.train_stride)
            self.train_data['ppg'].extend(w_ppg); self.train_data['temp'].extend(w_t); self.train_data['acc'].extend(w_a); self.train_data['label'].extend([label]*len(w_ppg))
            
            v_w_ppg, v_w_t, v_w_a = self._extract_windows(processed_ppg[train_len:past_len], processed_temp[train_len:past_len], processed_acc[:, train_len:past_len], self.train_stride)
            self.val_data['ppg'].extend(v_w_ppg); self.val_data['temp'].extend(v_w_t); self.val_data['acc'].extend(v_w_a); self.val_data['label'].extend([label]*len(v_w_ppg))
            
            # --- [2] Enrollment Session (5분, 25초 안정기 제외, Window_size 자르기) ---
            e_start = past_len
            e_end = past_len + enroll_len
            drop_len = 25 * self.fs 
            
            e_w_ppg, e_w_t, e_w_a = self._extract_windows(
                processed_ppg[e_start + drop_len : e_end], 
                processed_temp[e_start + drop_len : e_end], 
                processed_acc[:, e_start + drop_len : e_end], 
                stride=self.window_size # No Overlap
            )
            self.enroll_data[label] = {'ppg': e_w_ppg, 'temp': e_w_t, 'acc': e_w_a}

            # --- [3] Test Sessions (3분 * 10개) ---
            t_start = e_end
            session_len = 3 * 60 * self.fs
            self.test_data[label] = {}
            
            for s in range(10):
                s_start = t_start + s * session_len
                s_valid_start = s_start + drop_len # 25초 Drop
                
                # 남은 155초에서 Overlap 없이 10개의 윈도우 추출
                s_w_ppg, s_w_t, s_w_a = [], [], []
                for w_idx in range(10):
                    w_start = s_valid_start + (w_idx * self.window_size)
                    w_end = w_start + self.window_size
                    s_w_ppg.append(processed_ppg[w_start:w_end])
                    s_w_t.append(processed_temp[w_start:w_end])
                    s_w_a.append(processed_acc[:, w_start:w_end])

                self.test_data[label][s] = {'ppg': s_w_ppg, 'temp': s_w_t, 'acc': s_w_a}
                
    def _extract_windows(self, ppg, temp, acc, stride):
        w_ppg, w_temp, w_acc = [], [], []
        num_windows = (len(ppg) - self.window_size) // stride + 1
        for i in range(num_windows):
            start = i * stride
            end = start + self.window_size
            w_ppg.append(ppg[start:end])
            w_temp.append(temp[start:end])
            w_acc.append(acc[:, start:end])
        return w_ppg, w_temp, w_acc

class SplittedDataset(Dataset):
    def __init__(self, data_dict):
        self.ppg = np.expand_dims(np.array(data_dict['ppg'], dtype=np.float32), 1)
        self.temp = np.expand_dims(np.array(data_dict['temp'], dtype=np.float32), 1)
        self.acc = np.array(data_dict['acc'], dtype=np.float32)
        self.labels = np.array(data_dict['label'], dtype=np.int64)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.ppg[idx], self.temp[idx], self.acc[idx], self.labels[idx]

# ==============================================================================
# 2. Model Architecture: [1D ResNet + MLP + MHA]
# ==============================================================================
class AAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=32.0, m=0.35):
        super().__init__()
        self.s, self.m, self.weight = s, m, nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = torch.where(cosine > self.th, cosine * self.cos_m - sine * self.sin_m, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)

class ResNet1DEncoder(nn.Module):
    def __init__(self, in_channels, d_model=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResBlock1D(64, 64, stride=1)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, d_model, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.pool(x).squeeze(-1) # (B, d_model)

class TempMLP(nn.Module):
    def __init__(self, seq_len=512, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        return self.net(x.squeeze(1)) # (B, d_model)

class MultiModalMHANet(nn.Module):
    def __init__(self, num_users=12, d_model=128, seq_len=512):
        super().__init__()
        # 1. Modality Tokenizers
        self.ppg_enc = ResNet1DEncoder(in_channels=1, d_model=d_model)
        self.acc_enc = ResNet1DEncoder(in_channels=3, d_model=d_model)
        self.temp_enc = TempMLP(seq_len=seq_len, d_model=d_model)
        
        # 2. Multi-Head Attention Fusion
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 3. Projector & Classifier
        self.projector = nn.Sequential(
            nn.Linear(3 * d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256)
        )
        self.aam_softmax = AAMSoftmax(256, num_users, s=32.0, m=0.35)

    def forward(self, x_p, x_t, x_a, labels=None):
        emb_p = self.ppg_enc(x_p)
        emb_a = self.acc_enc(x_a)
        emb_t = self.temp_enc(x_t)
        
        tokens = torch.stack([emb_p, emb_a, emb_t], dim=1) # (B, 3, d_model)
        attn_out, _ = self.mha(tokens, tokens, tokens)
        fused_tokens = self.layer_norm(tokens + attn_out)  # (B, 3, d_model)
        
        flat_feat = fused_tokens.view(fused_tokens.size(0), -1) # (B, 384)
        
        final_emb = self.projector(flat_feat)
        final_emb = F.normalize(final_emb, p=2, dim=1)
        
        if labels is not None:
            return self.aam_softmax(final_emb, labels), final_emb
        return final_emb

# ==============================================================================
# 3. Training Loop & Validation Evaluation
# ==============================================================================
def get_val_eer(model, val_loader, device):
    """ Val 앞 5분으로 분산 Top 5 템플릿 생성 후 나머지로 EER 계산 """
    model.eval()
    all_embs, all_labels, all_noises = [], [], []
    
    with torch.no_grad():
        for p, t, a, labels in val_loader:
            p, t, a = p.to(device), t.to(device), a.to(device)
            emb = model(p, t, a) 
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(labels.numpy())
            # ACC 3축 분산 계산
            noise = torch.sum(torch.var(a.cpu(), dim=2), dim=1).numpy()
            all_noises.extend(noise)
            
    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.array(all_labels)
    all_noises = np.array(all_noises)
    
    unique_users = np.unique(all_labels)
    centroids = {}
    test_indices = []
    
    # 4초 윈도우, 2초 슬라이드에서 5분은 149개 윈도우
    enroll_pool_size = 149 
    
    for u in unique_users:
        u_idx = np.where(all_labels == u)[0]
        if len(u_idx) <= enroll_pool_size: continue
            
        pool_idx = u_idx[:enroll_pool_size] 
        test_idx = u_idx[enroll_pool_size:] 
        test_indices.extend(test_idx) 
        
        # 분산 가장 적은 상위 5개 선별
        k = min(5, len(pool_idx))
        pool_noises = all_noises[pool_idx]
        best_local_idx = np.argsort(pool_noises)[:k] 
        best_global_idx = pool_idx[best_local_idx]   
        
        cent = np.mean(all_embs[best_global_idx], axis=0)
        centroids[u] = cent / (np.linalg.norm(cent) + 1e-8)
        
    scores, true_labels = [], []
    for i in test_indices:
        emb = all_embs[i]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        true_u = all_labels[i]
        
        for u, cent in centroids.items():
            sim = np.dot(emb, cent) 
            scores.append(sim)
            true_labels.append(1 if true_u == u else 0)
            
    if len(scores) == 0: return 1.0 
        
    fpr, tpr, _ = roc_curve(true_labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx]

def train_model(model, train_loader, val_loader, device, epochs=50, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_eer = float('inf')
    early_stop_counter = 0
    best_model_path = 'best_mha_net.pth'

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for p, t, a, labels in progress_bar:
            p, t, a, labels = p.to(device), t.to(device), a.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, emb = model(p, t, a, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                raw_logits = F.linear(emb, F.normalize(model.aam_softmax.weight))
                preds = torch.argmax(raw_logits, dim=1)
                correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for p, t, a, labels in val_loader:
                p, t, a, labels = p.to(device), t.to(device), a.to(device), labels.to(device)
                emb = model(p, t, a)
                raw_logits = F.linear(emb, F.normalize(model.aam_softmax.weight))
                preds = torch.argmax(raw_logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_eer = get_val_eer(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val EER: {val_eer*100:.2f}%")
        
        if val_eer < best_eer:
            best_eer = val_eer
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  🌟 Best Model Saved! (Val EER: {best_eer*100:.2f}%)")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"🛑 Early stopping triggered!")
                break

    return best_model_path

# ==============================================================================
# 4. 실시간 검증 평가 (Enrollment QC & Test 10 Sessions)
# ==============================================================================
def get_embeddings(model, data_dict, device):
    model.eval()
    ppg = torch.FloatTensor(np.expand_dims(data_dict['ppg'], 1)).to(device)
    temp = torch.FloatTensor(np.expand_dims(data_dict['temp'], 1)).to(device)
    acc = torch.FloatTensor(np.array(data_dict['acc'])).to(device)
    with torch.no_grad():
        embs = model(ppg, temp, acc)
    return embs.cpu().numpy()

def evaluate_realtime_verification(model, splitter, num_users, device):
    print("\n--- 🔍 Enrollment Quality Control & Template Generation ---")
    templates = {}
    
    for u in range(num_users):
        if u in splitter.enroll_data:
            acc_windows = np.array(splitter.enroll_data[u]['acc'])
            # 각 후보 윈도우 ACC 3축 분산 계산 및 합산
            variances = np.sum(np.var(acc_windows, axis=2), axis=1) 
            
            # 노이즈가 가장 적은 상위 5개 선별
            best_indices = np.argsort(variances)[:5] 
            embs = get_embeddings(model, splitter.enroll_data[u], device)
            clean_embs = embs[best_indices]
            template = np.mean(clean_embs, axis=0) 
            templates[u] = template / (np.linalg.norm(template) + 1e-8)

    print("\n--- 📊 Session-by-Session Test Evaluation ---")
    session_eers, session_accs = [], []
    
    for s in range(10): 
        gen_scores, imp_scores = [], []
        
        for true_u in range(num_users):
            if true_u not in splitter.test_data or s not in splitter.test_data[true_u]:
                continue
            
            curr_embs = get_embeddings(model, splitter.test_data[true_u][s], device)
            
            for emb in curr_embs:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                
                # 등록한 Template과 코사인 유사도 비교
                for target_u, template in templates.items():
                    sim = np.dot(emb, template) 
                    if target_u == true_u: gen_scores.append(sim)
                    else: imp_scores.append(sim)
        
        if len(gen_scores) > 0 and len(imp_scores) > 0:
            y_true = np.array([1]*len(gen_scores) + [0]*len(imp_scores))
            y_score = np.array(gen_scores + imp_scores)
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
            eer = fpr[eer_idx]
            session_eers.append(eer)
            
            optimal_thresh = thresholds[eer_idx]
            y_pred = (y_score >= optimal_thresh).astype(int)
            acc = accuracy_score(y_true, y_pred)
            session_accs.append(acc)
            
            print(f"Session {s+1:<2} | EER: {eer*100:.2f}% | Accuracy: {acc*100:.2f}%")

    print("="*50)
    print(f"🏆 Final Average EER: {np.mean(session_eers)*100:.2f}%")
    print(f"🎯 Final Average Acc: {np.mean(session_accs)*100:.2f}%")
    print("="*50)

# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = "/Data/CRS25/PPG_Certifiation/data/Final_Data"
    num_users = 12

    splitter = BiometricDataSplitter(data_folder)
    
    train_loader = DataLoader(SplittedDataset(splitter.train_data), batch_size=64, shuffle=True)
    val_loader = DataLoader(SplittedDataset(splitter.val_data), batch_size=64, shuffle=False)
    
    # 1D ResNet + MLP + MHA 기반 모델 호출
    model = MultiModalMHANet(num_users=num_users).to(device)
    
    best_model_path = train_model(model, train_loader, val_loader, device)
    
    model.load_state_dict(torch.load(best_model_path))
    evaluate_realtime_verification(model, splitter, num_users, device)