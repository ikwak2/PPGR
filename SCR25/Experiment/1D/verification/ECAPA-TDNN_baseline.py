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

NUM_USERS = 16
FS = 128

WINDOW_SIZE = 128 * 4     # 4초
STRIDE = WINDOW_SIZE // 2   # 50% Overlap 
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
EMBED_DIM = 192

print(f"🚀 [Real-time Verification] ECAPA-TDNN + AAMSoftmax + Session-based Test")
print(f"⚙️  25s Stabilization Drop | Window: {WINDOW_SIZE} | Stride: {STRIDE}")

# ==========================================
# 2. Dataset (Session-based Split & 25s Drop)
# ==========================================
class TriModalDataset(Dataset):
    def __init__(self, data_folder, mode='train', train_val_ratio=0.8):
        self.mode = mode
        self.window_size = WINDOW_SIZE
        self.stride = STRIDE
        self.fs = FS
        
        # [설계 반영] 시간 정의
        self.test_duration = 30 * 60 * self.fs      # 테스트 30분
        self.enroll_duration = 5 * 60 * self.fs     # 등록 5분
        self.session_duration = 3 * 60 * self.fs    # 1개 세션 3분
        self.drop_samples = 25 * self.fs            # 안정화 기간 25초 Drop
        self.windows_per_session = 10               # 세션당 추출할 윈도우 수 (Overlap X)

        self.ppg, self.temp, self.acc, self.labels = [], [], [], []
        self.session_ids = []

        files = sorted(glob.glob(os.path.join(data_folder, "user_*.csv")))
        assert len(files) > 0, "No data found"

        for fp in files:
            user_id = int(os.path.basename(fp).split('_')[1].split('.')[0]) - 1
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

            # --- Time-based Split Logic ---
            total_samples = len(processed_ppg)
            test_start_idx = total_samples - self.test_duration
            enroll_start_idx = test_start_idx - self.enroll_duration
            
            # 1. 학습용 Pool (과거 데이터)
            if self.mode in ['train', 'val']:
                pool_ppg = processed_ppg[:enroll_start_idx]
                pool_temp = processed_temp[:enroll_start_idx]
                pool_acc = processed_acc[:, :enroll_start_idx]
                
                split_idx = int(len(pool_ppg) * train_val_ratio)
                if self.mode == 'train':
                    u_ppg, u_temp, u_acc = pool_ppg[:split_idx], pool_temp[:split_idx], pool_acc[:, :split_idx]
                else: 
                    u_ppg, u_temp, u_acc = pool_ppg[split_idx:], pool_temp[split_idx:], pool_acc[:, split_idx:]

                # Train/Val은 Overlap(Stride) 적용
                n_win = (len(u_ppg) - self.window_size) // self.stride
                for i in range(n_win):
                    start = i * self.stride
                    self._add_sample(u_ppg, u_temp, u_acc, start, user_id, session_id=0)

            # 2. Enrollment (템플릿 등록용 5분 - Quality Control 적용)
            elif self.mode == 'enroll':
                u_ppg = processed_ppg[enroll_start_idx:test_start_idx]
                u_temp = processed_temp[enroll_start_idx:test_start_idx]
                u_acc = processed_acc[:, enroll_start_idx:test_start_idx]
                
                # 25초(안정화 기간) 이후부터 사용 가능한 전체 구간 확인
                base_start = self.drop_samples
                valid_length = len(u_ppg) - base_start
                num_candidates = valid_length // self.window_size
                
                candidates = []
                
                # Step 1: 겹치지 않는 모든 윈도우 후보에 대해 모션 노이즈(분산) 계산
                for i in range(num_candidates):
                    start = base_start + (i * self.window_size)
                    end = start + self.window_size
                    
                    # 해당 윈도우의 가속도 데이터 (3, window_size)
                    acc_window = u_acc[:, start:end]
                    
                    # 가속도 3축 각각의 분산(Variance)을 구한 뒤 합산 = 움직임의 척도
                    motion_noise = np.sum(np.var(acc_window, axis=1))
                    
                    candidates.append((start, motion_noise))
                
                # Step 2: 모션 노이즈(분산)가 가장 작은 순서대로 오름차순 정렬
                candidates.sort(key=lambda x: x[1])
                
                # Step 3: 가장 움직임이 적었던 최상위 10개의 윈도우 시작점 추출
                best_starts = [cand[0] for cand in candidates[:self.windows_per_session]]
                
                # (선택 사항) 시간 순서대로 다시 정렬하여 자연스러운 흐름 유지
                best_starts.sort()
                
                # Step 4: 선별된 10개의 Clean 윈도우만 Enrollment 데이터로 추가
                for start in best_starts:
                    self._add_sample(u_ppg, u_temp, u_acc, start, user_id, session_id=0) # user_id 로 적용

            # 3. Test Sessions (실시간 인증 평가용 30분)
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
    def __init__(self, in_channels, channels=[256, 256, 256, 256, 768], lin_neurons=192):
        super().__init__()
        self.layer1 = TDNNBlock(in_channels, channels[0], 5, 1)
        self.layer2 = Res2NetBlock(channels[0], channels[1], dilation=2)
        self.layer3 = Res2NetBlock(channels[1], channels[2], dilation=3)
        self.layer4 = Res2NetBlock(channels[2], channels[3], dilation=4)
        self.layer5 = TDNNBlock(channels[1]*3, channels[4], 1, 1)
        self.asp = AttentiveStatisticsPooling(channels[4])
        self.bn_asp = nn.BatchNorm1d(channels[4] * 2)
        self.fc = nn.Linear(channels[4] * 2, lin_neurons)
        self.bn_final = nn.BatchNorm1d(lin_neurons)
    def forward(self, x):
        x1 = self.layer1(x)
        x2, x3, x4 = self.layer2(x1), self.layer3(self.layer2(x1)), self.layer4(self.layer3(self.layer2(x1)))
        x5 = self.layer5(torch.cat((x2, x3, x4), dim=1))
        x_pool = self.bn_asp(self.asp(x5).unsqueeze(2)).squeeze(2)
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
    def __init__(self, num_users=16, embed_dim=192):
        super().__init__()
        self.ppg_enc = ECAPA_TDNN_1D(1, lin_neurons=embed_dim)
        self.tmp_enc = ECAPA_TDNN_1D(1, lin_neurons=embed_dim)
        self.acc_enc = ECAPA_TDNN_1D(3, lin_neurons=embed_dim)
        self.projector = nn.Sequential(nn.Linear(embed_dim*3, embed_dim), nn.BatchNorm1d(embed_dim))
    def forward(self, ppg, temp, acc):
        e = torch.cat([self.ppg_enc(ppg), self.tmp_enc(temp), self.acc_enc(acc)], dim=1)
        return self.projector(e)

# ==========================================
# 4. Evaluation Logic
# ==========================================
def calculate_eer(genuine_scores, impostor_scores):
    scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    far = np.array([np.sum(impostor_scores >= t) / len(impostor_scores) for t in thresholds])
    frr = np.array([np.sum(genuine_scores < t) / len(genuine_scores) for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    return (far[idx] + frr[idx]) / 2

def extract_embeddings(model, loader):
    model.eval()
    all_embs, all_lbls, all_sess = [], [], []
    with torch.no_grad():
        for (p, t, a), lbl, sess in loader:
            feat = model(p.to(DEVICE), t.to(DEVICE), a.to(DEVICE))
            feat = F.normalize(feat, p=2, dim=1) # Extraction 시점에 L2 Normalize 보장
            all_embs.append(feat.cpu().numpy())
            all_lbls.append(lbl.numpy())
            all_sess.append(sess.numpy())
    return np.concatenate(all_embs), np.concatenate(all_lbls), np.concatenate(all_sess)

def evaluate_val_eer(model, val_loader):
    embs, lbls, _ = extract_embeddings(model, val_loader)
    templates, gen_scores, imp_scores = {}, [], []
    
    for u in range(NUM_USERS):
        u_idx = np.where(lbls == u)[0]
        if len(u_idx) < 5: continue
        raw_template = np.mean(embs[u_idx[:5]], axis=0)
        templates[u] = raw_template / (np.linalg.norm(raw_template) + 1e-8)
        
        for i in u_idx[5:]:
            gen_scores.append(np.dot(embs[i], templates[u]))
            for other_u in range(NUM_USERS):
                if other_u != u and other_u in templates:
                    imp_scores.append(np.dot(embs[i], templates[other_u]))
                    
    return calculate_eer(np.array(gen_scores), np.array(imp_scores))

def evaluate_realtime_verification(model, enroll_loader, test_loader):
    en_embs, en_lbls, _ = extract_embeddings(model, enroll_loader)
    templates = {}
    for u in range(NUM_USERS):
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

    model = TriModalECAPAModel(num_users=NUM_USERS, embed_dim=EMBED_DIM).to(DEVICE)
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
        # Batch Size 1 (실시간 1개 윈도우 추론 가정)
        dummy_p = torch.randn(1, 1, window_size).to(device)
        dummy_t = torch.randn(1, 1, window_size).to(device)
        dummy_a = torch.randn(1, 3, window_size).to(device)

        # 1. 총 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())

        # 2. FLOPs 계산 (MACs)
        flops = 0
        if HAS_THOP:
            # labels 없이 forward 하도록 입력값 조정 (SE 모델은 labels=None 지원)
            flops, _ = profile(model, inputs=(dummy_p, dummy_t, dummy_a), verbose=False)

        # 3. 추론 시간 측정 (안정성을 위해 100회 반복 평균)
        with torch.no_grad():
            for _ in range(10): # Warm-up
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

    # 측정 함수 실행
    measure_efficiency(model, DEVICE, WINDOW_SIZE)

    # --- Training ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_val, correct, total = 0, 0, 0
        for (p, t, a), l, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            p, t, a, l = p.to(DEVICE), t.to(DEVICE), a.to(DEVICE), l.to(DEVICE)
            
            emb = model(p, t, a)
            loss = criterion(emb, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            
            # Train Accuracy 계산
            with torch.no_grad():
                weights_norm = F.normalize(criterion.weight, p=2, dim=1)
                logits = torch.mm(F.normalize(emb, p=2, dim=1), weights_norm.t()) * criterion.scale
                _, pred = torch.max(logits, 1)
                correct += (pred == l).sum().item()
                total += l.size(0)

        # --- Validation ---
        model.eval()
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
        val_eer = evaluate_val_eer(model, val_loader) * 100
        print(f"   Loss: {loss_val/len(train_loader):.4f} | Train Acc: {100*correct/total:.2f}% | Val Acc: {v_acc:.2f}% | Val EER: {val_eer:.2f}%")

    # --- 최종 평가 (Real-time Session Simulation) ---
    print("\n🏁 Final Evaluation on 10 Independent Sessions (30-min Test Set)...")
    avg_eer, session_eers = evaluate_realtime_verification(model, enroll_loader, test_loader)
    
    print("\n📊 [Session-by-Session EER Results]")
    for s, s_eer in enumerate(session_eers):
        print(f"   Session {s+1} (3min): {s_eer*100:.3f}%")
        
    print(f"\n🏆 Final Average EER (Robustness): {avg_eer*100:.4f}%")