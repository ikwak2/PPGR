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
# 1. Config & Exclusions
# ==========================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_USERS = 12
FS = 128

WINDOW_SIZE = 128 * 4       # 4초
STRIDE = WINDOW_SIZE // 2   # 50% Overlap 
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
EMBED_DIM = 192

# 🔥 암기(과적합)를 막기 위해 Dropout 강화
DROPOUT_RATE = 0.45
EARLY_STOPPING_PATIENCE = 8  

# 🚨 데이터 결측치 예외 처리 (Python 0-based Index)
# User 4 -> 0-based index 3 / User 6 -> 0-based index 5
EXCLUSIONS = {
    3: (3786939, 4194810), 
    5: (4337572, 4545543)  
}

print(f"🚀 [SCI Paper Model] Context-Driven Bounded Gating (In-Session Evaluation)")
print(f"⚙️  Pure Activity Protocol | Dropout: {DROPOUT_RATE} | Exclusions Active")

# ==========================================
# 2. Dataset (Pure Activity & Exclusion Handling)
# ==========================================
class TriModalDataset(Dataset):
    def __init__(self, data_folder, mode='train', train_val_ratio=0.8):
        self.mode = mode
        self.window_size = WINDOW_SIZE
        self.stride = STRIDE
        self.fs = FS
        
        self.min_to_samples = 60 * self.fs
        self.act_starts = [240, 300, 360, 420, 480] 
        self.act_names = ["Museum", "Elevator", "Lunch", "Snow Walk", "Indoor Rest"]

        self.ppg, self.temp, self.acc, self.labels, self.session_ids = [], [], [], [], []

        files = sorted(glob.glob(os.path.join(data_folder, "user_*.csv")))
        for fp in files:
            user_id = int(os.path.basename(fp).split('_')[1].split('.')[0]) - 1
            if user_id >= NUM_USERS: continue
                
            df = pd.read_csv(fp)
            raw_ppg = df['PPG'].values
            raw_temp = df['temperature'].values
            raw_acc = df[['acc_x', 'acc_y', 'acc_z']].values.T

            detrended = signal.detrend(raw_ppg)
            b, a = signal.butter(4, [0.5/(0.5*self.fs), 8.0/(0.5*self.fs)], btype='band')
            filtered_ppg = signal.filtfilt(b, a, detrended)
            processed_ppg = (filtered_ppg - filtered_ppg.mean()) / (filtered_ppg.std() + 1e-6)
            processed_temp = (raw_temp - 25.0) / (40.0 - 25.0)
            processed_acc = (raw_acc - raw_acc.mean(axis=1, keepdims=True)) / (raw_acc.std(axis=1, keepdims=True) + 1e-6)

            total_samples = len(processed_ppg)
            
            # --- Train / Val ---
            if self.mode in ['train', 'val']:
                for act_start in self.act_starts:
                    s_idx = (act_start + 10) * self.min_to_samples if act_start == 480 else act_start * self.min_to_samples
                    e_idx = (act_start + 50) * self.min_to_samples
                    if e_idx > total_samples: break
                    
                    pool_p = processed_ppg[s_idx:e_idx]
                    split_idx = int(len(pool_p) * train_val_ratio)
                    
                    offset = s_idx if self.mode == 'train' else s_idx + split_idx
                    u_ppg = processed_ppg[offset : s_idx + split_idx] if self.mode == 'train' else processed_ppg[offset : e_idx]
                    u_temp = processed_temp[offset : s_idx + split_idx] if self.mode == 'train' else processed_temp[offset : e_idx]
                    u_acc = processed_acc[:, offset : s_idx + split_idx] if self.mode == 'train' else processed_acc[:, offset : e_idx]

                    n_win = (len(u_ppg) - self.window_size) // self.stride
                    for i in range(n_win):
                        start = i * self.stride
                        abs_start, abs_end = offset + start, offset + start + self.window_size
                        
                        # 🚨 결측치 구간 스킵 로직
                        if user_id in EXCLUSIONS:
                            ex_s, ex_e = EXCLUSIONS[user_id]
                            if not (abs_end <= ex_s or abs_start >= ex_e): continue
                            
                        self._add_sample(u_ppg, u_temp, u_acc, start, user_id, 0)

            # --- Enrollment ---
            elif self.mode == 'enroll':
                s_idx, e_idx = 485 * self.min_to_samples, 490 * self.min_to_samples
                if e_idx <= total_samples:
                    base_start = 25 * self.fs
                    offset = s_idx + base_start
                    valid_length = (e_idx - s_idx) - base_start
                    
                    num_candidates = (valid_length - self.window_size) // self.stride + 1
                    candidates = []
                    for i in range(num_candidates):
                        start = base_start + (i * self.stride)
                        abs_start, abs_end = s_idx + start, s_idx + start + self.window_size
                        
                        if user_id in EXCLUSIONS:
                            ex_s, ex_e = EXCLUSIONS[user_id]
                            if not (abs_end <= ex_s or abs_start >= ex_e): continue
                            
                        end = start + self.window_size
                        motion_noise = np.sum(np.var(processed_acc[:, s_idx+start : s_idx+end], axis=1))
                        candidates.append((start, motion_noise))
                    
                    candidates.sort(key=lambda x: x[1])
                    for start in sorted([cand[0] for cand in candidates[:5]]):
                        self._add_sample(processed_ppg[s_idx:e_idx], processed_temp[s_idx:e_idx], processed_acc[:, s_idx:e_idx], start, user_id, 0)

            # --- Test ---
            elif self.mode == 'test':
                for act_idx, act_start in enumerate(self.act_starts):
                    s_idx, e_idx = (act_start + 50) * self.min_to_samples, (act_start + 60) * self.min_to_samples
                    if e_idx > total_samples: continue
                    
                    base_start = 25 * self.fs
                    valid_length = (e_idx - s_idx) - base_start
                    num_windows = valid_length // self.window_size
                    
                    for i in range(num_windows):
                        start = base_start + (i * self.window_size)
                        abs_start, abs_end = s_idx + start, s_idx + start + self.window_size
                        
                        if user_id in EXCLUSIONS:
                            ex_s, ex_e = EXCLUSIONS[user_id]
                            if not (abs_end <= ex_s or abs_start >= ex_e): continue
                            
                        self._add_sample(processed_ppg[s_idx:e_idx], processed_temp[s_idx:e_idx], processed_acc[:, s_idx:e_idx], start, user_id, act_idx)

        self.ppg = torch.tensor(np.array(self.ppg), dtype=torch.float32).unsqueeze(1)
        self.temp = torch.tensor(np.array(self.temp), dtype=torch.float32).unsqueeze(1)
        self.acc = torch.tensor(np.array(self.acc), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.session_ids = torch.tensor(self.session_ids, dtype=torch.long)
        print(f"[{self.mode.upper()}] samples loaded: {len(self.labels)}")

    def _add_sample(self, ppg, temp, acc, start, label, session_id):
        end = start + self.window_size
        self.ppg.append(ppg[start:end])
        self.temp.append(temp[start:end])
        self.acc.append(acc[:, start:end])
        self.labels.append(label)
        self.session_ids.append(session_id)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return (self.ppg[idx], self.temp[idx], self.acc[idx]), self.labels[idx], self.session_ids[idx]

# ==========================================
# 3. Model (Context-Driven Bounded Gating)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x): return x * self.fc(self.avg_pool(x).view(x.size(0), x.size(1))).view(x.size(0), x.size(1), 1)

class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding=None):
        super().__init__()
        if padding is None: padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x): return self.bn(F.relu(self.conv(x)))

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
        y, out = chunks[0], [chunks[0]]
        for i in range(self.scale - 1):
            y = F.relu(self.bns[i](self.convs[i](chunks[i+1] + (y if i > 0 else 0))))
            out.append(y)
        return self.se(torch.cat(out, dim=1)) + x

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.tdnn = nn.Conv1d(channels, attention_channels, 1)
        self.conv = nn.Conv1d(attention_channels, channels, 1)
    def forward(self, x):
        w = F.softmax(self.conv(torch.tanh(self.tdnn(x))), dim=2)
        mu = torch.sum(x * w, dim=2)
        var = torch.clamp(torch.sum((x**2) * w, dim=2) - mu**2, min=1e-7)
        return torch.cat((mu, torch.sqrt(var)), dim=1)

class ECAPA_TDNN_1D(nn.Module):
    def __init__(self, in_channels, channels=[256, 256, 256, 256, 768], lin_neurons=192, dropout_p=0.2):
        super().__init__()
        self.layer1 = TDNNBlock(in_channels, channels[0], 5, 1)
        self.layer2 = Res2NetBlock(channels[0], channels[1], dilation=2)
        self.layer3 = Res2NetBlock(channels[1], channels[2], dilation=3)
        self.layer4 = Res2NetBlock(channels[2], channels[3], dilation=4)
        self.layer5 = TDNNBlock(channels[1]*3, channels[4], 1, 1)
        self.asp = AttentiveStatisticsPooling(channels[4])
        self.bn_asp = nn.BatchNorm1d(channels[4] * 2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(channels[4] * 2, lin_neurons)
        self.bn_final = nn.BatchNorm1d(lin_neurons)
        
    def forward(self, x):
        x1 = self.layer1(x)
        x5 = self.layer5(torch.cat((self.layer2(x1), self.layer3(self.layer2(x1)), self.layer4(self.layer3(self.layer2(x1)))), dim=1))
        return self.bn_final(self.fc(self.dropout(self.bn_asp(self.asp(x5).unsqueeze(2)).squeeze(2))))

# 🌟 Bounded Gating 핵심 모듈
class PhysicalContextConditioning(nn.Module):
    def __init__(self, context_dim=8): # 차원 축소 (과적합 방지)
        super().__init__()
        self.acc_proj = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU(inplace=True))
        self.tmp_proj = nn.Sequential(nn.Linear(1, context_dim), nn.ReLU(inplace=True))
        self.gate_generator = nn.Sequential(nn.Linear(context_dim * 2, 3), nn.Sigmoid())

    def forward(self, raw_acc, temp):
        acc_noise_level = torch.mean(torch.var(raw_acc, dim=2), dim=1, keepdim=True)
        tmp_level = torch.mean(temp, dim=2) 
        
        ctx_combined = torch.cat([self.acc_proj(acc_noise_level), self.tmp_proj(tmp_level)], dim=1)
        gates = self.gate_generator(ctx_combined) 
        
        # 🌟 코사인 붕괴 방지: 게이트 최소값을 0.5로 강제 (Bounded)
        bounded_gates = 0.5 + (0.5 * gates)
        return bounded_gates[:, 0:1], bounded_gates[:, 1:2], bounded_gates[:, 2:3]

class TriModalECAPAModel(nn.Module):
    def __init__(self, num_users=12, embed_dim=192, dropout_p=0.45):
        super().__init__()
        self.ppg_enc = ECAPA_TDNN_1D(1, lin_neurons=embed_dim, dropout_p=dropout_p)
        self.tmp_enc = ECAPA_TDNN_1D(1, lin_neurons=embed_dim, dropout_p=dropout_p)
        self.acc_enc = ECAPA_TDNN_1D(3, lin_neurons=embed_dim, dropout_p=dropout_p)
        self.context_cond = PhysicalContextConditioning(context_dim=8)
        
        # 🌟 Concat 대신 Gating을 사용하므로, Projector 입력은 순정(192*3)과 동일
        self.projector = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), 
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(p=dropout_p)
        )
        
    def forward(self, ppg, temp, acc, return_features=False):
        p_emb, t_emb, a_emb = self.ppg_enc(ppg), self.tmp_enc(temp), self.acc_enc(acc)
        w_p, w_t, w_a = self.context_cond(acc, temp)
        
        # 🌟 환경 정보(맥락)를 임베딩에 직접 섞지 않고, 볼륨(가중치)으로만 곱해줌
        e = torch.cat([p_emb * w_p, t_emb * w_t, a_emb * w_a], dim=1)
        final_emb = self.projector(e)
        
        if return_features: return final_emb, p_emb, a_emb 
        return final_emb

class AAMSoftmax(nn.Module):
    def __init__(self, in_features, n_classes, margin=0.2, scale=30):
        super().__init__()
        self.margin, self.scale = margin, scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, labels):
        cosine = torch.clamp(F.linear(F.normalize(x), F.normalize(self.weight)), -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine**2)
        phi = torch.where(cosine > math.cos(math.pi - self.margin), cosine * math.cos(self.margin) - sine * math.sin(self.margin), cosine - math.sin(self.margin) * self.margin)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        return F.cross_entropy(((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale, labels)

# ==========================================
# 4. Evaluation & Main
# ==========================================
def calculate_eer_and_thresh(genuine_scores, impostor_scores):
    if len(genuine_scores) == 0 or len(impostor_scores) == 0: return 0.0, 0.0
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
            feat = F.normalize(model(p.to(DEVICE), t.to(DEVICE), a.to(DEVICE)), p=2, dim=1) 
            all_embs.append(feat.cpu().numpy())
            all_lbls.append(lbl.numpy())
            all_sess.append(sess.numpy())
    return np.concatenate(all_embs), np.concatenate(all_lbls), np.concatenate(all_sess)

def evaluate_realtime_verification(model, enroll_loader, test_loader, num_users, global_thresh):
    en_embs, en_lbls, _ = extract_embeddings(model, enroll_loader)
    templates = {u: np.mean(en_embs[np.where(en_lbls == u)[0]], axis=0) / (np.linalg.norm(np.mean(en_embs[np.where(en_lbls == u)[0]], axis=0)) + 1e-8) for u in range(num_users) if len(np.where(en_lbls == u)[0]) > 0}
        
    ts_embs, ts_lbls, ts_sess = extract_embeddings(model, test_loader)
    results = []
    act_names = ["Museum", "Elevator", "Lunch", "Snow Walk", "Indoor Rest"]
    
    for s in range(5): 
        s_idx = np.where(ts_sess == s)[0]
        if len(s_idx) == 0: continue
        curr_embs, curr_lbls = ts_embs[s_idx], ts_lbls[s_idx]
        
        gen_scores, imp_scores, gen_preds, imp_preds = [], [], [], []
        for i, emb in enumerate(curr_embs):
            true_u = curr_lbls[i]
            for u, template in templates.items():
                sim = np.dot(emb, template)
                if u == true_u: 
                    gen_scores.append(sim)
                    gen_preds.append(1 if sim >= global_thresh else 0)
                else: 
                    imp_scores.append(sim)
                    imp_preds.append(1 if sim >= global_thresh else 0)
                    
        eer, _ = calculate_eer_and_thresh(np.array(gen_scores), np.array(imp_scores))
        acc = (sum(gen_preds) + (len(imp_preds) - sum(imp_preds))) / (len(gen_preds) + len(imp_preds)) if (len(gen_preds) + len(imp_preds)) > 0 else 0
        results.append((act_names[s], eer, acc))
    return results

if __name__ == "__main__":
    data_root = "/Data/CRS25/PPG_Certifiation/data/Final_Data"
    
    train_loader = DataLoader(TriModalDataset(data_root, "train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TriModalDataset(data_root, "val"), batch_size=BATCH_SIZE)
    enroll_loader = DataLoader(TriModalDataset(data_root, "enroll"), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TriModalDataset(data_root, "test"), batch_size=BATCH_SIZE)

    model = TriModalECAPAModel(num_users=NUM_USERS, embed_dim=EMBED_DIM, dropout_p=DROPOUT_RATE).to(DEVICE)
    criterion = AAMSoftmax(EMBED_DIM, NUM_USERS).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=LR)

    best_val_eer = float('inf')
    patience_counter = 0
    saved_models_history = [] 

    for epoch in range(1, EPOCHS + 1):
        model.train() 
        loss_val, correct, total = 0, 0, 0
        for (p, t, a), l, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            p, t, a, l = p.to(DEVICE), t.to(DEVICE), a.to(DEVICE), l.to(DEVICE)
            emb = model(p, t, a)
            loss = criterion(emb, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            
            with torch.no_grad():
                logits = torch.mm(F.normalize(emb, p=2, dim=1), F.normalize(criterion.weight, p=2, dim=1).t()) * criterion.scale
                correct += (torch.max(logits, 1)[1] == l).sum().item()
                total += l.size(0)

        # Validation 추출
        embs, lbls, _ = extract_embeddings(model, val_loader)
        templates = {u: np.mean(embs[np.where(lbls == u)[0][:5]], axis=0) / (np.linalg.norm(np.mean(embs[np.where(lbls == u)[0][:5]], axis=0)) + 1e-8) for u in range(NUM_USERS) if len(np.where(lbls == u)[0]) >= 5}
        
        gen_scores, imp_scores = [], []
        for u in range(NUM_USERS):
            u_idx = np.where(lbls == u)[0]
            if len(u_idx) < 5: continue
            for i in u_idx[5:]:
                gen_scores.append(np.dot(embs[i], templates[u]))
                for other_u in templates:
                    if other_u != u: imp_scores.append(np.dot(embs[i], templates[other_u]))
                    
        val_eer_raw, val_thresh = calculate_eer_and_thresh(np.array(gen_scores), np.array(imp_scores))
        val_eer = val_eer_raw * 100
        
        print(f"   Loss: {loss_val/len(train_loader):.4f} | Train Acc: {100*correct/total:.2f}% | Val EER: {val_eer:.2f}% (Thresh: {val_thresh:.4f})")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_counter = 0
            best_model_path = f"best_DualGating_dropout_ep{epoch}_eer{val_eer:.2f}.pt"
            torch.save(model.state_dict(), best_model_path)
            saved_models_history.append({'epoch': epoch, 'path': best_model_path, 'thresh': val_thresh, 'val_eer': val_eer})
            print(f"   💾 [Model Saved] -> '{best_model_path}'")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch}! Best Val EER: {best_val_eer:.2f}%")
                break

    print(f"\n🏁 Final Evaluation on ALL Saved Best Models ({len(saved_models_history)} models)...")
    for model_info in saved_models_history:
        m_epoch, m_path, m_thresh = model_info['epoch'], model_info['path'], model_info['thresh']
        print(f"\n🚀 Evaluating Epoch {m_epoch} Model: {m_path} (Thresh: {m_thresh:.4f})")
        
        if os.path.exists(m_path):
            model.load_state_dict(torch.load(m_path, map_location=DEVICE))
            model.eval() 
            session_results = evaluate_realtime_verification(model, enroll_loader, test_loader, NUM_USERS, m_thresh)
            
            avg_eer = sum(r[1] for r in session_results) / len(session_results)
            avg_acc = sum(r[2] for r in session_results) / len(session_results)
            print(f"🏆 [Epoch {m_epoch} Final] In-Session Avg EER: {avg_eer*100:.4f}% | Avg Accuracy: {avg_acc*100:.4f}%")