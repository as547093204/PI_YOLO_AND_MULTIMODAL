import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks

from model import ImprovedFusionModel
from video_image_process import emvCoreColor

# ----------------- 配置 -----------------
WEIGHTS_PATH = "/your_own_YOLO_model_path.pt/"
FUSION_CKPT = '/your_own_iPPGsignalmodel.pth/'
BASELINE_EXCEL = 'your_own_path/ulcer_signal_0.xlsx'
VIDEO_PATH = 'your_own_path/02014 1期.mp4'

# 类别→颜色 映射 (BGR)
COLOR_MAP = {
    'stage1':       (0, 255,   0),
    'stage2':       (0, 165, 255),
    'stage3':       (0,   0, 255),
    'stage4':       (255, 0,   0),
    'stage5':       (255, 0, 255),
    'unstageable':  (0,   0,   0),
}

# ----------------- 辅助函数 -----------------
def rgb_to_yiq(pixel):
    r, g, b = pixel
    y = 0.299*r + 0.587*g + 0.114*b
    i = 0.596*r - 0.275*g - 0.321*b
    q = 0.212*r - 0.523*g + 0.311*b
    return [y, i, q]

def pad_or_truncate(signal, length=150):
    if len(signal) < length:
        return np.pad(signal, (0, length - len(signal)), 'wrap')
    return signal[:length]

def smooth_contours_with_spline(contours, num_points=300):
    curves = []
    for cnt in contours:
        pts = cnt.reshape(-1,2)
        if len(pts) < 3:
            curves.append(pts.astype(np.int32))
            continue
        x, y = pts[:,0], pts[:,1]
        tck, u = splprep([x, y], s=5.0, per=True)
        u2 = np.linspace(0,1,num_points)
        x2, y2 = splev(u2, tck)
        curves.append(np.vstack((x2, y2)).T.astype(np.int32))
    return curves

def calculate_pulse_wave_features(signal, fs):
    sig = np.array(signal)
    peaks, _ = find_peaks(sig)
    foots = find_peaks(-sig)[0]
    feats = dict.fromkeys(['PWHH','PA','A1','A2','PH','CT','PI','Tn','ΔT','DT'], np.nan)
    if peaks.size and foots.size:
        hh = (sig[peaks[0]] + sig[foots[0]]) / 2
        pts = np.where(sig >= hh)[0]
        if pts.size > 1:
            feats['PWHH'] = (pts[-1] - pts[0]) / fs
        if foots.size > 1:
            feats['PA'] = np.trapz(sig[foots[0]:foots[-1]], dx=1/fs)
            a1_start, a1_end = sorted((foots[0], peaks[0]))
            feats['A1'] = np.trapz(sig[a1_start:a1_end], dx=1/fs) if a1_end > a1_start else np.nan
            a2_start, a2_end = sorted((peaks[0], foots[-1]))
            feats['A2'] = np.trapz(sig[a2_start:a2_end], dx=1/fs) if a2_end > a2_start else np.nan
            feats['PH'] = sig[peaks[0]] - sig[foots[0]]
    return feats

def load_fusion_model(device):
    model = ImprovedFusionModel(temporal_length=150, num_classes=5).to(device)
    chk = torch.load(FUSION_CKPT, map_location=device)
    state = chk.get('model_state_dict', chk)
    md = model.state_dict()
    md.update({k: v for k, v in state.items() if k in md and v.size() == md[k].size()})
    model.load_state_dict(md)
    model.eval()
    return model

# ----------------- 主流程 -----------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 加载模型
    yolo = YOLO(WEIGHTS_PATH)
    yolo.to(device)
    fusion_model = load_fusion_model(device)

    # 2. 计算基线特征
    baseline = pd.read_excel(BASELINE_EXCEL).values
    fe_fs = 30
    base_feats = [calculate_pulse_wave_features(row, fe_fs) for row in baseline]
    baseline_means = {k: np.nanmean([d[k] for d in base_feats if not np.isnan(d[k])]) for k in base_feats[0]}

    # 3. 打开视频并读取首帧
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame0 = cap.read()
    if not ret:
        print("无法读取视频")
        return
    h0, w0 = frame0.shape[:2]

    # 4. YOLO 检测所有提案，基于中心点计算较小 ROI
    results = yolo(frame0, device=device, imgsz=640, conf=0.2)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names
    if scores.size == 0:
        print("YOLO 未检测到目标，退出")
        cap.release()
        return

    # 计算各框中心并聚合ROI
    centers = np.column_stack(((boxes[:,0] + boxes[:,2]) // 2,
                                (boxes[:,1] + boxes[:,3]) // 2))
    cx, cy = centers.mean(axis=0).astype(int)
    w_union = boxes[:,2].max() - boxes[:,0].min()
    h_union = boxes[:,3].max() - boxes[:,1].min()
    scale = 0.6
    roi_w, roi_h = int(w_union*scale), int(h_union*scale)
    x1 = max(0, cx - roi_w//2); y1 = max(0, cy - roi_h//2)
    x2 = min(w0, x1 + roi_w); y2 = min(h0, y1 + roi_h)

    # 可视化检测框 + 中心ROI
    vis = frame0.copy()
    for (b, s, c) in zip(boxes, scores, classes):
        bx1, by1, bx2, by2 = b
        color = COLOR_MAP.get(names[c], (255,255,255))
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(vis, f"{names[c]} {s:.2f}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 3)
    cv2.imshow("YOLO 中心ROI", cv2.resize(vis, (800,600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5. EMV 放大
    PAD = 50
    x_exp = max(0, x1 - PAD)
    y_exp = max(0, y1 - PAD)
    w_exp = min(w0 - x_exp, (x2 - x1) + 2 * PAD)
    h_exp = min(h0 - y_exp, (y2 - y1) + 2 * PAD)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(150):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    exp_roi_frames = [f[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp] for f in frames]
    amp_exp = emvCoreColor(
        exp_roi_frames, fps,
        maxLevel=6, freqLow=0.83/2, freqHigh=1.0/2,
        alpha=100.0, chromAttenuation=1.0,
        method="ideal"
    )
    pad_x = x1 - x_exp
    pad_y = y1 - y_exp
    amp_roi = [
        frame[pad_y:pad_y+(y2-y1), pad_x:pad_x+(x2-x1)] for frame in amp_exp
    ]

    # 6. 网格分类 & 可视化叠加
    grid = 10
    gh, gw = (y2-y1)//grid, (x2-x1)//grid
    scaler = StandardScaler()
    preds = np.zeros((grid, grid), int)
    non_zero = []
    for i in tqdm(range(grid), desc="网格分类"):
        for j in range(grid):
            y0g, x0g = i * gh, j * gw
            y1g, x1g = y0g + gh, x0g + gw
            series = [
                np.mean(
                    np.apply_along_axis(rgb_to_yiq, 2, amp_roi[t][y0g:y1g, x0g:x1g])[:, :, 0]
                )
                for t in range(len(amp_roi))
            ]
            arr = pad_or_truncate(series)
            norm = scaler.fit_transform(np.array(arr).reshape(-1, 1)).flatten()
            tnsr = torch.tensor(norm, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = fusion_model(tnsr.unsqueeze(0))
                _, cls = torch.max(out, 1)
            c = cls.item()
            preds[i, j] = c
            if c != 0:
                non_zero.append(norm)

    mask = np.zeros((y2-y1, x2-x1, 3), np.uint8)
    cmap = {0:(0,255,0),1:(0,0,255),2:(0,0,200),3:(0,0,150),4:(0,0,100)}
    for i in range(grid):
        for j in range(grid):
            if preds[i, j] != 0:
                cv2.rectangle(
                    mask,
                    (j*gw, i*gh),
                    ((j+1)*gw, (i+1)*gh),
                    cmap[preds[i, j]],
                    -1
                )
    overlay = cv2.addWeighted(amp_roi[0].astype(np.uint8), 0.7, mask, 0.3, 0)

    binm = np.zeros((y2-y1, x2-x1), np.uint8)
    for i in range(grid):
        for j in range(grid):
            if preds[i, j] != 0:
                binm[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = 255
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spl = smooth_contours_with_spline(cnts)
    poly = np.zeros_like(overlay)
    cv2.fillPoly(poly, spl, (0, 0, 100))
    overlay = cv2.addWeighted(overlay, 1.0, poly, 0.8, 0)
    for c in spl:
        cv2.polylines(overlay, [c], True, (0, 0, 100), 2)

    # 7. 结果展示
    result = frame0.copy()
    result[y1:y2, x1:x2] = overlay
    cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Result", cv2.resize(result, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 8. 打印分类占比及 IPPG 特征
    flat = preds.flatten()
    cnts_arr = np.bincount(flat, minlength=5)
    labels = ["无压疮","I期","II期","III期","IV期"]
    print("分类占比：")
    for idx, lab in enumerate(labels):
        print(f"  {lab}: {cnts_arr[idx]/flat.size*100:.2f}%")

    if non_zero:
        avg = np.mean(non_zero, axis=0)
        feats = calculate_pulse_wave_features(avg, fe_fs)
        print("\nIPPG 特征：")
        for k, v in feats.items():
            if not np.isnan(v):
                print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()
