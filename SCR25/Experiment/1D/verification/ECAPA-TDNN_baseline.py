import os, glob, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from tqdm.auto import tqdm
import time 

# ==========================================
# 1. Config
# ==========================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [수정 1] 16명 -> 12명으로 변경
NUM_USERS = 12
FS = 128

WINDOW_SIZE = 128 * 4     # 4초
STRIDE = WINDOW_SIZE // 2   # 50% Overlap 
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
EMBED_DIM = 192

# 🔥 Dropout 비율 추가
DROPOUT_RATE = 0.2

# [수정 4] Early Stopping 파라미터 추가
EARLY_STOPPING_PATIENCE = 5  

print(f"🚀 [Real-time Verification] ECAPA-TDNN + AAMSoftmax + Session-based Test")
print(f"⚙️  25s Stabilization Drop | Window: {WINDOW_SIZE} | Stride: {STRIDE} | Users: {NUM_USERS} | Dropout: {DROPOUT_RATE}")

# ==========================================
# 2. Dataset (Session-based Split & 25s Drop)
# ==========================================
class TriModalDataset(Dataset):
    def __init__(self, data_folder, mode='train', train_val_ratio=0.8):
        self.mode = mode
        self.window_size = WINDOW_SIZE
        self.stride = STRIDE
        self.fs = FS
        
        self.test_duration = 30 * 60 * self.fs      
        self.enroll_duration = 5 * 60 * self.fs     
        self.session_duration = 3 * 60 * self.fs    
        self.drop_samples = 25 * self.fs            
        self.windows_per_session = 10               

        self.ppg, self.temp, self.acc, self.labels = [], [], [], []
        self.session_ids = []

        files = sorted(glob.glob(os.path.join(data_folder, "user_*.csv")))
        assert len(files) > 0, "No data found"

        for fp in files:
            user_id = int(os.path.basename(fp).split('_')[1].split('.')[0]) - 1
            
            # [수정 1] 지정된 NUM_USERS(12명)까지만 데이터 로드
            if user_id >= NUM_USERS:
                continue
                
            df = pd.read_csv(fp)

            raw_ppg = df['PPG'].values
            raw_temp = df['temperature'].values
            raw_acc = df[['acc_x', 'acc_y', 'acc_z']].values.T

            # Preprocessing
            detrended = signal.detrend(raw_ppg)
            b, a = signal.butter(4, [0.5/(0.5*self.fs), 8.0/(0.5*self.fs)], btype='band')
            filtered_ppg = signal.filtfilt(b, a, detrended)
            processed_ppg = (filtered_ppg - filtered_ppg.mean()) / (filtered_ppg.std() + 1e-6)
            processed_temp = (raw_temp - 25.0) / (40.0 - 25.0)
            processed_acc = (raw_acc - raw_acc.mean(axis=1, keepdims=True)) / (raw_acc.std(axis=1, keepdims=True) + 1e-6)

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

                n_win = (len(u_ppg) - self.window_size) // self.stride
                for i in range(n_win):
                    start = i * self.stride
                    self._add_sample(u_ppg, u_temp, u_acc, start, user_id, session_id=0)

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
                    self._add_sample(u_ppg, u_temp, u_acc, start, user_id, session_id=0)

            elif self.mode == 'test':
                u_ppg = processed_ppg[test_start_idx:]
                u_temp = processed_temp[test_start_idx:]
                u_acc = processed_acc[:, test_start_idx:]
                
                for s in range(10):
                    sess_start_idx = s * self.session_duration
                    base_start = sess_start_idx + self.drop_samples
                    for i in range(self.windows_per_session):
                        start = base_start + (i * self.window_size)
                        self._add_sample(u_ppg, u_temp, u_acc, start, user_id, session_id=s)

        self.ppg = torch.tensor(np.array(self.ppg), dtype=torch.float32).unsqueeze(1)
        self.temp = torch.tensor(np.array(self.temp), dtype=torch.float32).unsqueeze(1)
        self.acc = torch.tensor(np.array(self.acc), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.session_ids = torch.tensor(self.session_ids, dtype=torch.long)
        print(f"[{self.mode.upper()}] samples: {len(self.labels)}")

    def _add_sample(self, ppg, temp, acc, start, label, session_id):
        end = start + self.window_size
        self.ppg.append(ppg[start:end])
        self.temp.append(temp[start:end])
        self.acc.append(acc[:, start:end])
        self.labels.append(label)
        self.session_ids.append(session_id)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.ppg[idx], self.temp[idx], self.acc[idx]), self.labels[idx], self.session_ids[idx]

# ==========================================
# 3. Model Components (ECAPA-TDNN)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding=None):
        super().__init__()
        if padding is None: padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x): return self.bn(self.relu(self.conv(x)))

class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, dilation=1):
        super().__init__()
        self.width = out_channels // scale
        self.scale = scale
        self.convs = nn.ModuleList([nn.Conv1d(self.width, self.width, 3, dilation=dilation, padding=dilation) for _ in range(scale - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        self.se = SEBlock(out_channels)
    def forward(self, x):
        chunks = torch.split(x, self.width, dim=1)
        y = chunks[0]
        out = [y]
        for i in range(self.scale - 1):
            y = self.convs[i](chunks[i+1] + (y if i > 0 else 0))
            y = F.relu(self.bns[i](y))
            out.append(y)
        return self.se(torch.cat(out, dim=1)) + x

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.tdnn = nn.Conv1d(channels, attention_channels, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, 1)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        w = self.softmax(self.conv(self.tanh(self.tdnn(x))))
        mu = torch.sum(x * w, dim=2)
        var = torch.clamp(torch.sum((x**2) * w, dim=2) - mu**2, min=1e-7)
        return torch.cat((mu, torch.sqrt(var)), dim=1)

class ECAPA_TDNN_1D(nn.Module):
    # 🔥 dropout_p 파라미터 추가
    def __init__(self, in_channels, channels=[256, 256, 256, 256, 768], lin_neurons=192, dropout_p=0.2):
        super().__init__()
        self.layer1 = TDNNBlock(in_channels, channels[0], 5, 1)
        self.layer2 = Res2NetBlock(channels[0], channels[1], dilation=2)
        self.layer3 = Res2NetBlock(channels[1], channels[2], dilation=3)
        self.layer4 = Res2NetBlock(channels[2], channels[3], dilation=4)
        self.layer5 = TDNNBlock(channels[1]*3, channels[4], 1, 1)
        self.asp = AttentiveStatisticsPooling(channels[4])
        self.bn_asp = nn.BatchNorm1d(channels[4] * 2)
        
        # 🔥 Dropout 레이어 추가 (ASP 이후 적용)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.fc = nn.Linear(channels[4] * 2, lin_neurons)
        self.bn_final = nn.BatchNorm1d(lin_neurons)
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2, x3, x4 = self.layer2(x1), self.layer3(self.layer2(x1)), self.layer4(self.layer3(self.layer2(x1)))
        x5 = self.layer5(torch.cat((x2, x3, x4), dim=1))
        
        x_pool = self.bn_asp(self.asp(x5).unsqueeze(2)).squeeze(2)
        x_pool = self.dropout(x_pool) # 🔥 Fully Connected 전에 Dropout 적용
        
        return self.bn_final(self.fc(x_pool))

class AAMSoftmax(nn.Module):
    def __init__(self, in_features, n_classes, margin=0.2, scale=30):
        super().__init__()
        self.margin, self.scale = margin, scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine**2)
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        phi = torch.where(cosine > math.cos(math.pi - self.margin), phi, cosine - math.sin(self.margin) * self.margin)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.scale, labels)

class TriModalECAPAModel(nn.Module):
    # 🔥 dropout_p 파라미터 추가
    def __init__(self, num_users=16, embed_dim=192, dropout_p=0.2):
        super().__init__()
        self.ppg_enc = ECAPA_TDNN_1D(1, lin_neurons=embed_dim, dropout_p=dropout_p)
        self.tmp_enc = ECAPA_TDNN_1D(1, lin_neurons=embed_dim, dropout_p=dropout_p)
        self.acc_enc = ECAPA_TDNN_1D(3, lin_neurons=embed_dim, dropout_p=dropout_p)
        
        # 🔥 Fusion Projector에 Dropout 레이어 추가
        self.projector = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim*3, embed_dim), 
            nn.BatchNorm1d(embed_dim)
        )
        
    def forward(self, ppg, temp, acc):
        e = torch.cat([self.ppg_enc(ppg), self.tmp_enc(temp), self.acc_enc(acc)], dim=1)
        return self.projector(e)

# ==========================================
# 4. Evaluation Logic
# ==========================================
def calculate_eer(genuine_scores, impostor_scores):
    if len(genuine_scores) == 0 or len(impostor_scores) == 0: return 0.0
    scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    far = np.array([np.sum(impostor_scores >= t) / len(impostor_scores) for t in thresholds])
    frr = np.array([np.sum(genuine_scores < t) / len(genuine_scores) for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    return (far[idx] + frr[idx]) / 2

def extract_embeddings(model, loader):
    model.eval() # 평가 모드로 전환 (Dropout 비활성화 됨)
    all_embs, all_lbls, all_sess = [], [], []
    with torch.no_grad():
        for (p, t, a), lbl, sess in loader:
            feat = model(p.to(DEVICE), t.to(DEVICE), a.to(DEVICE))
            feat = F.normalize(feat, p=2, dim=1) 
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
                    
    return calculate_eer(np.array(gen_scores), np.array(imp_scores))

def evaluate_realtime_verification(model, enroll_loader, test_loader, num_users):
    en_embs, en_lbls, _ = extract_embeddings(model, enroll_loader)
    templates = {}
    for u in range(num_users):
        u_idx = np.where(en_lbls == u)[0]
        if len(u_idx) == 0: continue
        raw_template = np.mean(en_embs[u_idx], axis=0)
        templates[u] = raw_template / (np.linalg.norm(raw_template) + 1e-8)
        
    ts_embs, ts_lbls, ts_sess = extract_embeddings(model, test_loader)
    session_eers = []
    
    for s in range(10): 
        s_idx = np.where(ts_sess == s)[0]
        if len(s_idx) == 0: continue
        curr_embs, curr_lbls = ts_embs[s_idx], ts_lbls[s_idx]
        
        gen_scores, imp_scores = [], []
        for i, emb in enumerate(curr_embs):
            true_u = curr_lbls[i]
            for u, template in templates.items():
                sim = np.dot(emb, template)
                if u == true_u: gen_scores.append(sim)
                else: imp_scores.append(sim)
                
        eer = calculate_eer(np.array(gen_scores), np.array(imp_scores))
        session_eers.append(eer)
        
    return np.mean(session_eers), session_eers

# ==========================================
# 5. Main Pipeline
# ==========================================
if __name__ == "__main__":
    data_root = "/Data/CRS25/PPG_Certifiation/data/Final_Data"
    
    train_loader = DataLoader(TriModalDataset(data_root, "train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TriModalDataset(data_root, "val"), batch_size=BATCH_SIZE)
    enroll_loader = DataLoader(TriModalDataset(data_root, "enroll"), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TriModalDataset(data_root, "test"), batch_size=BATCH_SIZE)

    # 🔥 모델 선언 시 Dropout Rate 전달
    model = TriModalECAPAModel(num_users=NUM_USERS, embed_dim=EMBED_DIM, dropout_p=DROPOUT_RATE).to(DEVICE)
    criterion = AAMSoftmax(EMBED_DIM, NUM_USERS).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=LR)

    try:
        from thop import profile
        HAS_THOP = True
    except ImportError:
        HAS_THOP = False
        print("💡 [Tip] FLOPs 측정을 위해 thop를 설치해주세요: pip install thop")

    def measure_efficiency(model, device, window_size):
        model.eval()
        dummy_p = torch.randn(1, 1, window_size).to(device)
        dummy_t = torch.randn(1, 1, window_size).to(device)
        dummy_a = torch.randn(1, 3, window_size).to(device)

        total_params = sum(p.numel() for p in model.parameters())

        flops = 0
        if HAS_THOP:
            flops, _ = profile(model, inputs=(dummy_p, dummy_t, dummy_a), verbose=False)

        with torch.no_grad():
            for _ in range(10): 
                _ = model(dummy_p, dummy_t, dummy_a)
                
            start_time = time.time()
            for _ in range(100):
                _ = model(dummy_p, dummy_t, dummy_a)
            end_time = time.time()
            
        avg_time_ms = ((end_time - start_time) / 100) * 1000

        print("\n" + "="*55)
        print("🚀 [Model Efficiency Report]")
        print(f"   Total Parameters : {total_params / 1e6:.4f} M (Million)")
        if HAS_THOP:
            print(f"   Computational Cost: {flops / 1e6:.4f} MFLOPs")
        print(f"   Inference Time   : {avg_time_ms:.4f} ms / window")
        print("="*55 + "\n")

    measure_efficiency(model, DEVICE, WINDOW_SIZE)

    # [수정 4 & 5] Early Stopping 관련 변수
    best_val_eer = float('inf')
    patience_counter = 0
    best_model_path = "best_ecapa_model.pt"

    # --- Training ---
    for epoch in range(1, EPOCHS + 1):
        model.train() # 모델 학습 모드 전환 (Dropout 활성화 됨)
        loss_val, correct, total = 0, 0, 0
        for (p, t, a), l, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            p, t, a, l = p.to(DEVICE), t.to(DEVICE), a.to(DEVICE), l.to(DEVICE)
            
            emb = model(p, t, a)
            loss = criterion(emb, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            
            # [수정 2] Train Accuracy 계산을 명시적으로 반영
            with torch.no_grad():
                weights_norm = F.normalize(criterion.weight, p=2, dim=1)
                logits = torch.mm(F.normalize(emb, p=2, dim=1), weights_norm.t()) * criterion.scale
                _, pred = torch.max(logits, 1)
                correct += (pred == l).sum().item()
                total += l.size(0)

        # [수정 2] Train Acc 백분율 연산
        train_acc = 100 * correct / total

        # --- Validation ---
        model.eval() # 모델 검증 모드 전환 (Dropout 비활성화 됨)
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for (p, t, a), l, _ in val_loader:
                p, t, a, l = p.to(DEVICE), t.to(DEVICE), a.to(DEVICE), l.to(DEVICE)
                emb = model(p, t, a)
                
                weights_norm = F.normalize(criterion.weight, p=2, dim=1)
                logits = torch.mm(F.normalize(emb, p=2, dim=1), weights_norm.t()) * criterion.scale
                _, pred = torch.max(logits, 1)
                v_correct += (pred == l).sum().item()
                v_total += l.size(0)
                
        v_acc = 100 * v_correct / v_total
        val_eer = evaluate_val_eer(model, val_loader, NUM_USERS) * 100
        
        # [수정 2] 출력문에 Train Acc 표시
        print(f"   Loss: {loss_val/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {v_acc:.2f}% | Val EER: {val_eer:.2f}%")

        # [수정 4 & 5] Early Stopping 체크 및 모델 저장
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   [Model Saved] Best Val EER updated to {best_val_eer:.2f}%!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}! Best Val EER was {best_val_eer:.2f}%")
                break

    # --- 최종 평가 (Real-time Session Simulation) ---
    print("\n🏁 Final Evaluation on 10 Independent Sessions (30-min Test Set)...")
    
    # [수정 5] 저장해둔 최적 모델 가중치 로드
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval() # 평가 모드 전환 (Dropout 비활성화 됨)

    # [수정 3] Final Evaluation 소요 시간 측정 시작
    eval_start_time = time.time()
    
    avg_eer, session_eers = evaluate_realtime_verification(model, enroll_loader, test_loader, NUM_USERS)
    
    # [수정 3] 종료 및 소요시간 연산
    eval_duration = time.time() - eval_start_time
    
    print("\n📊 [Session-by-Session EER Results]")
    for s, s_eer in enumerate(session_eers):
        print(f"   Session {s+1} (3min): {s_eer*100:.3f}%")
        
    print(f"\n🏆 Final Average EER (Robustness): {avg_eer*100:.4f}%")
    # [수정 3] 소요 시간 출력
    print(f"⏱️ Evaluation Time Taken: {eval_duration:.2f} seconds")