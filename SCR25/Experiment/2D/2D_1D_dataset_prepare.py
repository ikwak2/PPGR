# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import cv2
from scipy import signal # 필터링(butter/filtfilt)은 작동하므로 유지
from tqdm.auto import tqdm

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
DATA_FOLDER = "/Data/CRS25/PPG_Certifiation/data/Final_Data"
OUTPUT_FOLDER = "./processed_data_10sec"

FS = 128              # 샘플링 레이트
WINDOW_SEC = 10       # 10초
WINDOW_SIZE = FS * WINDOW_SEC # 1280
STRIDE = 128          # 1초 단위 이동

# CWT 관련 설정 (Morlet)
F_MIN = 0.5           # 최소 주파수 (Hz)
F_MAX = 8.0           # 최대 주파수 (Hz)
NUM_SCALES = 64       # 주파수 해상도
IMG_SIZE = (224, 224) # ResNet 입력 크기
W0 = 6                # Morlet Wavelet Omega0

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ---------------------------------------------------------
# [Manual Implementation] Numpy로 직접 구현한 CWT 함수들
# ---------------------------------------------------------
def manual_morlet_wavelet(M, s, w=6.0):
    """
    Generate complex Morlet wavelet
    M: length of the wavelet
    s: scale
    w: omega0
    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    # Morlet formula (Normalized)
    output = np.pi**(-0.25) * np.sqrt(1/s) * np.exp(-0.5 * x**2) * np.exp(1j * w * x)
    return output

def manual_cwt(data, scales, w=6.0):
    """
    Compute CWT using numpy convolution with correct cropping
    """
    # 결과 담을 배열 (Scale 개수 x 데이터 길이)
    output = np.zeros((len(scales), len(data)), dtype=np.complex128)
    
    for i, s in enumerate(scales):
        # 1. 웨이블릿 길이 결정 (저주파면 길어짐)
        M = int(10 * s) 
        if M % 2 == 0: M += 1 # 길이를 홀수로 맞춤
        
        # 2. 웨이블릿 생성
        wavelet = manual_morlet_wavelet(M, s, w)
        
        # 3. 컨볼루션 (Full mode)
        # Full 모드는 길이 = (N + M - 1)
        conv_full = np.convolve(data, wavelet, mode='full')
        
        # 4. 중앙 부분 자르기 (Cropping)
        # 웨이블릿의 중심이 신호의 각 지점과 매칭되는 구간만 추출
        # (len(wavelet) - 1) // 2 인덱스부터 시작하면 위상(Phase)이 맞음
        start_idx = (len(wavelet) - 1) // 2
        end_idx = start_idx + len(data)
        
        # 슬라이싱 범위를 데이터 길이에 딱 맞춤
        conv_cropped = conv_full[start_idx : end_idx]
        
        # 만약 계산 오차로 1~2개 차이나는 경우 방어 코드
        if len(conv_cropped) > len(data):
            conv_cropped = conv_cropped[:len(data)]
        elif len(conv_cropped) < len(data):
            # 부족하면 뒤에 0 채움 (거의 발생 안 함)
            padding = np.zeros(len(data) - len(conv_cropped), dtype=np.complex128)
            conv_cropped = np.concatenate([conv_cropped, padding])
            
        output[i, :] = conv_cropped
        
    return output

# ---------------------------------------------------------

def generate_cwt_image(sig, fs, f_min, f_max, num_scales, img_size):
    """
    PPG 신호를 CWT 스케일로그램 이미지(RGB)로 변환
    """
    # 1. 스케일 계산 (수동)
    freqs = np.linspace(f_min, f_max, num_scales)
    scales = (W0 * fs) / (2 * np.pi * freqs)
    
    # 2. CWT 수행 (수동 구현 함수 호출)
    cwt_mat = manual_cwt(sig, scales, w=W0)
    
    # 3. 절대값(Magnitude) 및 로그 스케일링
    cwt_abs = np.abs(cwt_mat)
    cwt_log = np.log1p(cwt_abs) 
    
    # 4. 정규화 (0~255)
    cwt_min = cwt_log.min()
    cwt_max = cwt_log.max()
    if cwt_max - cwt_min < 1e-6:
        cwt_norm = np.zeros_like(cwt_log)
    else:
        cwt_norm = (cwt_log - cwt_min) / (cwt_max - cwt_min)
        
    cwt_uint8 = (cwt_norm * 255).astype(np.uint8)
    
    # 5. 리사이징 (224x224)
    # cv2.resize는 (width, height) 순서임
    cwt_resized = cv2.resize(cwt_uint8, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    
    # 6. 컬러맵 적용 (Jet) -> RGB 변환
    cwt_color = cv2.applyColorMap(cwt_resized, cv2.COLORMAP_JET)
    cwt_rgb = cv2.cvtColor(cwt_color, cv2.COLOR_BGR2RGB)
    
    return cwt_rgb

def preprocess_all():
    print(f"🚀 전처리 시작: 윈도우 {WINDOW_SEC}초, CWT {F_MIN}-{F_MAX}Hz (Manual & Cropped)")
    
    search_pattern = os.path.join(DATA_FOLDER, "user_*.csv")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print("❌ 데이터 파일이 없습니다.")
        return

    img_save_dir = os.path.join(OUTPUT_FOLDER, "images")
    npy_save_dir = os.path.join(OUTPUT_FOLDER, "signals")
    create_directory(img_save_dir)
    create_directory(npy_save_dir)
    
    metadata = [] 
    sample_count = 0

    for filepath in tqdm(file_list, desc="Processing Users"):
        filename = os.path.basename(filepath)
        try:
            user_num = int(filename.split('_')[1].split('.')[0])
        except:
            continue
        
        label = user_num - 1 
        
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]
        
        # 사용자별 데이터 분할
        if user_num == 4:
            df_segments = [df.iloc[:3786928], df.iloc[4194811:]]
        elif user_num == 6:
            df_segments = [df.iloc[:4337569], df.iloc[4545544:]]
        else:
            df_segments = [df]
            
        for segment_df in df_segments:
            if segment_df.empty: continue
            
            if not all(col in segment_df.columns for col in ['PPG', 'temperature', 'acc_x', 'acc_y', 'acc_z']):
                continue

            raw_ppg = segment_df['PPG'].values
            raw_temp = segment_df['temperature'].values
            raw_acc = segment_df[['acc_x', 'acc_y', 'acc_z']].values 

            # 필터링
            try:
                b, a = signal.butter(4, [0.5/(0.5*FS), 8.0/(0.5*FS)], btype='band')
                ppg_filtered = signal.filtfilt(b, a, raw_ppg)
            except Exception as e:
                print(f"Filtering skipped due to error: {e}")
                ppg_filtered = raw_ppg 

            # 정규화
            temp_norm = (raw_temp - 25.0) / (40.0 - 25.0)
            acc_mean = np.mean(raw_acc, axis=0)
            acc_std = np.std(raw_acc, axis=0) + 1e-6
            acc_norm = (raw_acc - acc_mean) / acc_std

            num_windows = (len(ppg_filtered) - WINDOW_SIZE) // STRIDE
            
            for i in range(num_windows):
                start = i * STRIDE
                end = start + WINDOW_SIZE
                
                seg_ppg = ppg_filtered[start:end]
                seg_temp = temp_norm[start:end]
                seg_acc = acc_norm[start:end, :] 
                
                sample_id = f"user{user_num:02d}_{sample_count:07d}"
                
                try:
                    # A. PPG -> Manual CWT Image
                    cwt_img = generate_cwt_image(seg_ppg, FS, F_MIN, F_MAX, NUM_SCALES, IMG_SIZE)
                    img_path = os.path.join(img_save_dir, f"{sample_id}.png")
                    cv2.imwrite(img_path, cv2.cvtColor(cwt_img, cv2.COLOR_RGB2BGR))
                    
                    # B. Temp/Acc -> NPY (Temp: 1ch, Acc: 3ch -> Total 4ch)
                    combined_signal = np.hstack([seg_temp.reshape(-1, 1), seg_acc])
                    npy_path = os.path.join(npy_save_dir, f"{sample_id}.npy")
                    np.save(npy_path, combined_signal)
                    
                    metadata.append([sample_id, label])
                    sample_count += 1
                except Exception as e:
                    print(f"Error processing {sample_id}: {e}")
                    # 하나라도 실패하면 메타데이터에 안 넣고 건너뜀
                    continue

    df_meta = pd.DataFrame(metadata, columns=['sample_id', 'label'])
    df_meta.to_csv(os.path.join(OUTPUT_FOLDER, "metadata.csv"), index=False)
    
    print(f"\n🎉 전처리 완료! 총 샘플 수: {len(df_meta)}")
    print(f"📁 저장 위치: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    preprocess_all()