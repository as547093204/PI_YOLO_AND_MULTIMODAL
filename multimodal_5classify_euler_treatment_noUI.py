import cv2
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm

from model import ImprovedFusionModel
from video_image_process import emvCoreColor  # 确保此模块包含 emvCoreColor 函数
import os
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false'
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'GSTREAMER,V4L2,FFMPEG'
os.environ['OPENCV_GUI_BACKEND'] = 'GTK'  # 使用 GTK 作为 GUI 后端

# ————— 辅助函数 ————— #

def rgb_to_yiq(pixel):
    r, g, b = pixel
    y = 0.299*r + 0.587*g + 0.114*b
    i = 0.596*r - 0.275*g - 0.321*b
    q = 0.212*r - 0.523*g + 0.311*b
    return [y, i, q]

def pad_or_truncate(signal, length=150):
    if len(signal) < length:
        return np.pad(signal, (0, length - len(signal)), 'wrap')
    else:
        return signal[:length]

def load_model_checkpoint(model, checkpoint_path, device):
    chk = torch.load(checkpoint_path, map_location=device)
    state = chk.get('model_state_dict', chk)
    mdict = model.state_dict()
    filtered = {k:v for k,v in state.items() if k in mdict and v.size()==mdict[k].size()}
    mdict.update(filtered)
    model.load_state_dict(mdict)
    return model

def smooth_contours_with_spline(contours, num_points=300):
    from scipy.interpolate import splprep, splev
    curves = []
    for cnt in contours:
        pts = cnt.reshape(-1,2)
        if len(pts)<3:
            curves.append(pts.astype(np.int32))
            continue
        x,y = pts[:,0], pts[:,1]
        tck,u = splprep([x,y], s=5.0, per=True)
        u2 = np.linspace(0,1,num_points)
        x2,y2 = splev(u2, tck)
        curve = np.vstack((x2,y2)).T.astype(np.int32)
        curves.append(curve)
    return curves

def calculate_pulse_wave_features(signal, fs):
    from scipy.signal import find_peaks
    sig = np.array(signal)
    peaks,_ = find_peaks(sig)
    foots = find_peaks(-sig)[0]
    feats = dict.fromkeys(['PWHH','PA','A1','A2','PH','CT','PI','Tn','ΔT','DT'], np.nan)
    if len(peaks)>0 and len(foots)>0:
        # 半高宽 PWHH
        hh = (sig[peaks[0]] + sig[foots[0]])/2
        pts = np.where(sig>=hh)[0]
        if len(pts)>1:
            feats['PWHH'] = (pts[-1]-pts[0]) / fs
        # 面积 PA, A1, A2
        if len(foots)>1:
            feats['PA'] = np.trapz(sig[foots[0]:foots[-1]], dx=1/fs)
            a1_start,a1_end = sorted((foots[0],peaks[0]))
            feats['A1'] = np.trapz(sig[a1_start:a1_end], dx=1/fs) if a1_end>a1_start else np.nan
            a2_start,a2_end = sorted((peaks[0],foots[-1]))
            feats['A2'] = np.trapz(sig[a2_start:a2_end], dx=1/fs) if a2_end>a2_start else np.nan
            feats['PH'] = sig[peaks[0]] - sig[foots[0]]
    return feats

class FeatureExtractor:
    def __init__(self, sampling_rate=30):
        self.fs = sampling_rate

# ————— 主程序 ————— #

def main():
    # ——— 一：设备与模型 ———
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fusion_model = ImprovedFusionModel(temporal_length=150, num_classes=5).to(device)
    fusion_model = load_model_checkpoint(
        fusion_model,
        'models/mutimodal/best_model_20241209_221736.pth',
        device
    )
    fusion_model.eval()

    # ——— 二：计算基线 IPPG 特征 ———
    baseline_data = pd.read_excel('data/ulcer_signal_0.xlsx').values
    fe = FeatureExtractor(sampling_rate=30)
    base_feats = []
    for row in baseline_data:
        f = calculate_pulse_wave_features(row, fe.fs)
        if not all(np.isnan(v) for v in f.values()):
            base_feats.append(f)
    if base_feats:
        baseline_means = {k: np.nanmean([d[k] for d in base_feats]) for k in base_feats[0]}
    else:
        print("警告：无法计算基线特征，全部设为0")
        baseline_means = {k:0 for k in ['PWHH','PA','A1','A2','PH','CT','PI','Tn','ΔT','DT']}

    # ——— 三：读取视频 & 让用户在 640×480 窗口中选 ROI ———
    video_path = 'E:/amplify/input/hx_24_10_08/02014 1期.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        print("无法读取视频")
        return

    orig_h, orig_w = frame0.shape[:2]

    # 把第一帧缩放到 640×480，用于选择 ROI
    TARGET_WIN_W, TARGET_WIN_H = 640, 480
    frame_small = cv2.resize(frame0, (TARGET_WIN_W, TARGET_WIN_H), interpolation=cv2.INTER_AREA)

    x_s, y_s, w_s, h_s = cv2.selectROI("Select ROI (640×480)", frame_small, False)
    cv2.destroyAllWindows()
    if w_s == 0 or h_s == 0:
        print("未选定 ROI，程序结束。")
        cap.release()
        return

    # 将小图坐标映射回原始分辨率
    scale_x = orig_w / TARGET_WIN_W
    scale_y = orig_h / TARGET_WIN_H
    x0 = int(x_s * scale_x)
    y0 = int(y_s * scale_y)
    w0 = int(w_s * scale_x)
    h0 = int(h_s * scale_y)

    # ——— 四：定义“外扩” ROI 的区域，但保留 x0,y0,w0,h0 作为“模型使用的原始 ROI” ———
    ROI_PAD = 50  # 向外扩展 50 像素（可根据需求自行调整）
    # 计算外扩后 ROI 在原始帧上的坐标
    x_exp = max(0, x0 - ROI_PAD)
    y_exp = max(0, y0 - ROI_PAD)
    w_exp = min(orig_w - x_exp, w0 + 2 * ROI_PAD)
    h_exp = min(orig_h - y_exp, h0 + 2 * ROI_PAD)
    # 此时：(x_exp, y_exp, w_exp, h_exp) 为“放大用的外扩 ROI”，
    # 而 (x0, y0, w0, h0) 一直是不变的“模型计算用的原始 ROI”。

    # ——— 五：抽取前 150 帧 → 使用“外扩 ROI”进行 Eulerian 放大 ———
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(150):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 150:
        print("帧不足150，退出")
        return

    # 提取外扩 ROI 区的帧序列
    roi_exp_frames = [f[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp] for f in frames]

    # 对外扩 ROI 区做 Eulerian 放大
    amplified_exp_roi = emvCoreColor(
        roi_exp_frames, fps,
        maxLevel=6,
        freqLow=0.83/2, freqHigh=1.0/2,
        alpha=100.0, chromAttenuation=1.0,
        method="ideal"
    )
    # amplified_exp_roi 是一个列表，长度与 roi_exp_frames 一致，每帧大小为 (h_exp, w_exp)

    # ——— 六：从“放大后的外扩 ROI”中裁出“原始 ROI” → 用于网格分类 ———
    # 原始 ROI 在外扩 ROI 内的偏移量为 ROI_PAD 像素
    pad = ROI_PAD
    # 对于边界情况，如果外扩时到达图像边缘，则 pad 可能需要修正：
    # 真实偏移 pad_x = x0 - x_exp； pad_y = y0 - y_exp
    pad_x = x0 - x_exp
    pad_y = y0 - y_exp

    # 生成“放大后原始 ROI”（h0×w0） 的序列
    amplified_roi = [
        amp_frame[pad_y:pad_y+h0, pad_x:pad_x+w0]
        for amp_frame in amplified_exp_roi
    ]
    # amplified_roi 列表里每个元素的 shape 都是 (h0, w0, 3)

    # ——— 七：网格分类只针对“放大后原始 ROI” amp_roi 进行（尺寸 h0×w0） ———
    grid = 10
    sh, sw = h0 // grid, w0 // grid
    mask = np.zeros((h0, w0, 3), dtype=np.uint8)
    scaler = StandardScaler()
    preds = np.zeros((grid, grid), dtype=int)
    non_zero = []
    total_cells = grid * grid
    pbar = tqdm(total=total_cells, desc="分析进度")

    for i in range(grid):
        for j in range(grid):
            y0g, x0g = i * sh, j * sw
            y1g, x1g = min(y0g + sh, h0), min(x0g + sw, w0)
            # 计算放大后 ROI 每个网格块的 Y 通道平均
            series = [
                np.mean(
                    np.apply_along_axis(rgb_to_yiq, 2, amplified_roi[t][y0g:y1g, x0g:x1g])[:, :, 0]
                )
                for t in range(len(amplified_roi))
            ]
            arr = pad_or_truncate(series)
            norm = scaler.fit_transform(np.array(arr).reshape(-1, 1)).flatten()
            tnsr = torch.tensor(norm, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = fusion_model(tnsr.unsqueeze(0))
                _, cls = torch.max(out, 1)
            c = cls.item()
            preds[i, j] = c
            color_map = {
                0: (0, 255, 0),
                1: (0, 0, 255),
                2: (0, 0, 200),
                3: (0, 0, 150),
                4: (0, 0, 100)
            }
            mask[y0g:y1g, x0g:x1g] = color_map.get(c, (128, 128, 128))
            if c != 0:
                non_zero.append(norm)
            pbar.update(1)
    pbar.close()

    # ——— 八：构造叠加 & 平滑轮廓 ———
    alpha = 0.3
    # 先把 amplified_roi[0] 转为 uint8，与 mask 保持同样 dtype
    amp0_uint8 = amplified_roi[0].astype(np.uint8)
    # 再进行 addWeighted，这时两张图都是 uint8
    overlay_roi = cv2.addWeighted(amp0_uint8, 1 - alpha, mask, alpha, 0)

    # 后面平滑轮廓、叠加多边形等都基于 overlay_roi
    binm = np.zeros((h0, w0), dtype=np.uint8)
    for i in range(grid):
        for j in range(grid):
            if preds[i, j] != 0:
                y0g, x0g = i * sh, j * sw
                y1g, x1g = min(y0g + sh, h0), min(x0g + sw, w0)
                binm[y0g:y1g, x0g:x1g] = 255

    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spl = smooth_contours_with_spline(cnts)
    poly = np.zeros_like(overlay_roi)
    cv2.fillPoly(poly, spl, (0, 0, 100))
    overlay_roi = cv2.addWeighted(overlay_roi, 1.0, poly, 0.8, 0)
    for c in spl:
        cv2.polylines(overlay_roi, [c], True, (0, 0, 100), 2)

    # ——— 九：将“处理后的放大 ROI”贴回到原始第一帧上并显示 ———
    result = frame0.copy()
    result[y0:y0+h0, x0:x0+w0] = overlay_roi

    # ——— 十：打印分类占比 ———
    flat = preds.flatten()
    cnts_arr = np.bincount(flat, minlength=5)
    tot = cnts_arr.sum()
    names = ["0:无压疮", "1:I期", "2:II期", "3:III期", "4:IV期"]
    print("分类占比：")
    for idx, name in enumerate(names):
        print(f"  {name} → {cnts_arr[idx]/tot*100:.2f}%")

    # ——— 十一：计算并打印 IPPG 特征对比 ———
    if non_zero:
        avg = np.mean(non_zero, axis=0)
        feats = calculate_pulse_wave_features(avg, fe.fs)
        valid_feats = {k: v for k, v in feats.items() if not np.isnan(v)}
        if valid_feats:
            print("\nROI 平均 IPPG 特征：")
            suggestions = []
            for k, v in valid_feats.items():
                print(f"  {k}: {v:.4f}")

            if abs(valid_feats['PA']) > abs(baseline_means['PA']) * 2:
                suggestions.append("PA显著增高：提示局部血流阻力增加，建议改善局部血液循环（如按摩、促进血液回流）")
            elif abs(valid_feats['PA']) < abs(baseline_means['PA']) * 0.1:
                suggestions.append("PA显著降低：提示局部血流灌注不足，建议检查是否存在阻塞性压迫（如卧床时长过久）")
            else:
                suggestions.append("PA无显著异常：维持现有护理措施即可")

            if abs(valid_feats['PH']) > abs(baseline_means['PH']) * 2:
                suggestions.append("PH显著增高：提示局部压力较大，建议减压（如更频繁翻身、使用减压垫）")
            elif abs(valid_feats['PH']) < abs(baseline_means['PH']) * 0.1:
                suggestions.append("PH显著降低：提示局部循环动力不足，建议促进局部血管扩张（如温水擦拭）")
            else:
                suggestions.append("PH无显著异常：局部压力状态正常")

            if abs(valid_feats['A1']) < abs(baseline_means['A1']) * 0.1:
                suggestions.append("A1明显下降：心脏射血或局部供血受限，建议适当营养支持或局部理疗")
            elif abs(valid_feats['A1']) > abs(baseline_means['A1']) * 2:
                suggestions.append("A1明显增高：心脏射血或局部供血增强，建议皮肤冷敷或干燥避免感染炎症")
            else:
                suggestions.append("A1无显著异常：心脏射血与局部供血基本正常")

            if abs(valid_feats['PWHH']) > abs(baseline_means['PWHH']) * 2:
                suggestions.append("PWHH增宽：血管弹性变差，建议适度锻炼和血管健康管理")
            elif abs(valid_feats['PWHH']) < abs(baseline_means['PWHH']) * 0.1:
                suggestions.append("PWHH缩窄：血流速度增快，建议检查是否存在异常动脉狭窄或栓塞等")
            else:
                suggestions.append("PWHH无显著异常：血管弹性状况稳定")

            if abs(valid_feats['A2']) < abs(baseline_means['A2']) * 0.1:
                suggestions.append("A2降低：动脉顺应性下降，建议对皮肤和血液循环加强护理")
            elif abs(valid_feats['A2']) > abs(baseline_means['A2']) * 2:
                suggestions.append("A2增高：动脉顺应性增加，代偿性血流量增加，建议如必要监测血流动态，调整护理频率")
            else:
                suggestions.append("A2无显著异常：动脉顺应性保持良好")

            print("\n综合建议：")
            for s in suggestions:
                print("- " + s)
        else:
            print("\nROI 区域未能提取到有效 IPPG 特征。")

    # ——— 十二：显示最终结果（窗口固定 640×480） ———
    WINDOW_NAME = "Final Result (640×480)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    TARGET_SHOW_W, TARGET_SHOW_H = 640, 480
    cv2.resizeWindow(WINDOW_NAME, TARGET_SHOW_W, TARGET_SHOW_H)

    # 缩放 result 到 800×600 再显示
    show_img = cv2.resize(result, (TARGET_SHOW_W, TARGET_SHOW_H), interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
