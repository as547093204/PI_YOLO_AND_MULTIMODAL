import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import ImprovedFusionModel
from scipy.interpolate import splprep, splev  # 用于样条插值

class SignalDataset(Dataset):
    def __init__(self, signals, labels, augment=False):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.augment:
            if np.random.random() < 0.5:
                noise = torch.randn(signal.size()) * 0.05
                signal = signal + noise
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                signal = signal * scale
            if np.random.random() < 0.5:
                shift = np.random.randint(-10, 10)
                signal = torch.roll(signal, shift)

        return signal, label

def rgb_to_yiq(pixel):
    r, g, b = pixel
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b
    return [y, i, q]

def pad_or_truncate(signal, length=150):
    if len(signal) < length:
        return np.pad(signal, (0, length - len(signal)), 'wrap')
    else:
        return signal[:length]

def load_model_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_state_dict = model.state_dict()

    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.size() == model_state_dict[k].size():
                filtered_state_dict[k] = v
            else:
                print(f"Skipping loading parameter '{k}' due to size mismatch.")
        else:
            print(f"Skipping loading parameter '{k}' as it does not exist in the current model.")

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

    missing_keys = [k for k in model_state_dict.keys() if k not in filtered_state_dict]
    if missing_keys:
        print(f"Warning: The following keys were not loaded from the checkpoint: {missing_keys}")

    return model

def smooth_contours_with_spline(contours, num_points=200):
    smooth_curves = []
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        x = cnt[:,0]
        y = cnt[:,1]
        tck, u = splprep([x, y], s=5.0, per=True)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        curve = np.vstack((x_new, y_new)).T.astype(np.int32)
        smooth_curves.append(curve)
    return smooth_curves

class FeatureExtractor:
    def __init__(self, sampling_rate=30):
        self.sampling_rate = sampling_rate

    def find_peaks_valleys(self, signal):
        signal_np = signal.detach().cpu().numpy()
        peaks = []
        valleys = []
        for i in range(1, len(signal_np) - 1):
            if signal_np[i - 1] < signal_np[i] and signal_np[i] > signal_np[i + 1]:
                peaks.append(i)
            if signal_np[i - 1] > signal_np[i] and signal_np[i] < signal_np[i + 1]:
                valleys.append(i)
        return peaks, valleys

    def find_half_height_points(self, signal, peak_idx, valley_idx):
        peak_height = signal[peak_idx]
        valley_height = signal[valley_idx]
        half_height = (peak_height + valley_height) / 2

        left_idx = valley_idx
        right_idx = peak_idx

        for i in range(valley_idx, peak_idx):
            if signal[i] >= half_height:
                left_idx = i
                break

        for i in range(peak_idx, valley_idx, -1):
            if signal[i] >= half_height:
                right_idx = i
                break
        return left_idx, right_idx

    def compute_ippg_features(self, signal):
        peaks, valleys = self.find_peaks_valleys(signal)
        if len(peaks) < 1 or len(valleys) < 2:
            return None
        PA = torch.trapz(signal[valleys[0]:valleys[-1]], dx=1 / self.sampling_rate)
        A2 = torch.trapz(signal[peaks[0]:valleys[1]], dx=1 / self.sampling_rate)
        PH = signal[peaks[0]] - signal[valleys[0]]
        A1 = torch.trapz(signal[valleys[0]:peaks[0]], dx=1 / self.sampling_rate)
        left_idx, right_idx = self.find_half_height_points(signal, peaks[0], valleys[0])
        PWHH = (right_idx - left_idx) / self.sampling_rate

        return {
            'PA': PA.item(),
            'A2': A2.item(),
            'PH': PH.item(),
            'A1': A1.item(),
            'PWHH': PWHH
        }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = ImprovedFusionModel(temporal_length=150, num_classes=5).to(device)
    checkpoint_path = 'models/mutimodal/best_model_20241209_221736.pth'  # 替换为你的模型路径
    model = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    # 加载基线数据
    baseline_data = pd.read_excel('data/ulcer_signal_0.xlsx').values
    fe = FeatureExtractor(sampling_rate=30)
    baseline_features_list = []
    for row in baseline_data:
        sig_t = torch.tensor(row, dtype=torch.float32)
        fvals = fe.compute_ippg_features(sig_t)
        if fvals is not None:
            baseline_features_list.append(fvals)

    if len(baseline_features_list) > 0:
        baseline_means = {}
        keys = baseline_features_list[0].keys()
        for k in keys:
            baseline_means[k] = np.mean([d[k] for d in baseline_features_list])
    else:
        baseline_means = {'PA':0,'A2':0,'PH':0,'A1':0,'PWHH':0}
        print("警告：无法从ulcer_signal_0中计算基线特征。")

    video_path = 'E:/amplify/output/02013 2期_rebuild.mp4'  # 替换为你的视频路径
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频")
        cap.release()
        exit()

    roi = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]

    grid_size = 10
    subregion_height = h // grid_size
    subregion_width = w // grid_size

    mask = np.zeros_like(roi_frame)
    scaler = StandardScaler()
    predictions = np.zeros((grid_size, grid_size))

    class_colors = {
        0: [0, 255, 0],
        1: [0, 0, 255],
        2: [0, 0, 200],
        3: [0, 0, 150],
        4: [0, 0, 100]
    }

    total_subregions = grid_size * grid_size
    progress_bar = tqdm(total=total_subregions, desc="Processing Subregions")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(150):
        ret, f_ = cap.read()
        if not ret:
            break
        frames.append(f_)

    non_zero_signals = []

    for k in range(grid_size):
        for l in range(grid_size):
            start_y = k * subregion_height
            start_x = l * subregion_width
            end_y = min(start_y + subregion_height, h)
            end_x = min(start_x + subregion_width, w)

            region_y_channel_data = np.zeros((end_y - start_y, end_x - start_x, len(frames)))
            for i, f_ in enumerate(frames):
                roi_frame_current = f_[y:y + h, x:x + w]
                yiq_frame = np.apply_along_axis(rgb_to_yiq, 2, roi_frame_current[start_y:end_y, start_x:end_x])
                y_channel = yiq_frame[:, :, 0]
                region_y_channel_data[:, :, i] = y_channel

            mean_signal = np.mean(region_y_channel_data, axis=(0, 1))
            padded_signal = pad_or_truncate(mean_signal)
            padded_signal = scaler.fit_transform(padded_signal.reshape(-1, 1)).flatten()
            signal_tensor = torch.tensor(padded_signal, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(signal_tensor.unsqueeze(0))
                _, predicted = torch.max(output.data, 1)
                predicted_class = predicted.item()
                predictions[k, l] = predicted_class

            if predicted_class in class_colors:
                mask[start_y:end_y, start_x:end_x] = class_colors[predicted_class]
            else:
                mask[start_y:end_y, start_x:end_x] = [128, 128, 128]

            if predicted_class != 0:
                non_zero_signals.append(signal_tensor.cpu().numpy())

            progress_bar.update(1)

    progress_bar.close()
    cap.release()

    alpha = 0.2
    overlay_frame = cv2.addWeighted(roi_frame, 1 - alpha, mask, alpha, 0)

    # 创建二值掩码（非0类为255）
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    for k in range(grid_size):
        for l in range(grid_size):
            predicted_class = predictions[k, l]
            if predicted_class != 0:
                start_y = k * subregion_height
                start_x = l * subregion_width
                end_y = min(start_y + subregion_height, h)
                end_x = min(start_x + subregion_width, w)
                binary_mask[start_y:end_y, start_x:end_x] = 255

    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth_curves = smooth_contours_with_spline(contours, num_points=300)
    contour_overlay = overlay_frame.copy()
    fill_color = (0, 0, 100)
    cv2.fillPoly(contour_overlay, smooth_curves, fill_color)
    fill_alpha = 0.8
    overlay_frame = cv2.addWeighted(overlay_frame, 1.0, contour_overlay, fill_alpha, 0)
    for curve in smooth_curves:
        cv2.polylines(overlay_frame, [curve], True, fill_color, 2)

    frame[y:y + h, x:x + w] = overlay_frame

    # 分析预测结果
    all_predictions = predictions.flatten()
    class_counts = np.bincount(all_predictions.astype(int), minlength=5)
    total = class_counts.sum()

    stage_names = ["无压疮(0类)", "I期(1类)", "II期(2类)", "III期(3类)", "IV期(4类)"]
    for i in range(5):
        ratio = (class_counts[i] / total) * 100 if total > 0 else 0.0
        print(f"{stage_names[i]} 占比: {ratio:.2f}%")

    # 计算非0类区域平均信号的IPPG特征
    if len(non_zero_signals) > 0:
        avg_nonzero_signal = np.mean(non_zero_signals, axis=0)
        avg_nonzero_signal_t = torch.tensor(avg_nonzero_signal, dtype=torch.float32)
        curr_features = fe.compute_ippg_features(avg_nonzero_signal_t)
        if curr_features is not None:
            print("\n当前非0类区域平均IPPG特征值与基线对比:")
            suggestions = []
            for k,v in curr_features.items():
                base_val = abs(baseline_means[k])
                val_base = abs(v)
                print(f"{k}: {val_base:.4f} (基线:{base_val:.4f})")

            # 全部指标建议同时输出
            # 以下为简单逻辑示例，可以根据实际需求调整
            # 判断倍数偏差或百分比偏差
            PA_diff = v_ratio(curr_features['PA'], baseline_means['PA'])
            PH_diff = v_ratio(curr_features['PH'], baseline_means['PH'])
            A1_diff = v_ratio(curr_features['A1'], baseline_means['A1'])
            PWHH_diff = v_ratio(curr_features['PWHH'], baseline_means['PWHH'])
            A2_diff = v_ratio(curr_features['A2'], baseline_means['A2'])

            # 根据偏差阈值给出建议
            # 举例: 如果当前值 > 基线 * 2 认为显著增高, < 基线 * 0.1认为显著降低
            # 同时将所有建议加入suggestions列表中

            if abs(curr_features['PA']) > abs(baseline_means['PA']) * 2:
                suggestions.append("PA显著增高：提示局部血流阻力增加，建议改善局部血液循环（如按摩、促进血液回流）")
            elif abs(curr_features['PA']) < abs(baseline_means['PA']) * 0.1:
                suggestions.append("PA显著降低：提示局部血流灌注不足，建议检查是否存在阻塞性压迫（如卧床时长过久）")
            else:
                suggestions.append("PA无显著异常：维持现有护理措施即可")

            if abs(curr_features['PH']) > abs(baseline_means['PH'])* 2:
                suggestions.append("PH显著增高：提示局部压力较大，建议减压（如更频繁翻身、使用减压垫）")
            elif abs(curr_features['PH']) < abs(baseline_means['PH']) * 0.1:
                suggestions.append("PH显著降低：提示局部循环动力不足，建议促进局部血管扩张（如温水擦拭）")
            else:
                suggestions.append("PH无显著异常：局部压力状态正常")

            if abs(curr_features['A1']) < abs(baseline_means['A1']) * 0.1:
                suggestions.append("A1明显下降：心脏射血或局部供血受限，建议适当营养支持或局部理疗")
            elif abs(curr_features['A1']) > abs(baseline_means['A1']) * 2:
                suggestions.append("A1明显增高：心脏射血或局部供血增强，建议皮肤冷敷或干燥避免感染炎症")
            else:
                suggestions.append("A1无显著异常：心脏射血与局部供血基本正常")

            if abs(curr_features['PWHH']) > abs(baseline_means['PWHH']) * 2:
                suggestions.append("PWHH增宽：血管弹性变差，建议适度锻炼和血管健康管理")
            elif abs(curr_features['PWHH']) < abs(baseline_means['PWHH']) * 0.1:
                suggestions.append("PWHH缩窄：血流速度增快，建议检查是否存在异常动脉狭窄或栓赛等")
            else:
                suggestions.append("PWHH无显著异常：血管弹性状况稳定")

            if abs(curr_features['A2']) < abs(baseline_means['A2']) * 0.1:
                suggestions.append("A2降低：动脉顺应性下降，建议对皮肤和血液循环加强护理")
            elif abs(curr_features['A2']) > abs(baseline_means['A2']) * 2:
                suggestions.append("A2增高：动脉顺应性增加，代偿性血流量增加，建议如必要监测血流动态，调整护理频率")
            else:
                suggestions.append("A2无显著异常：动脉顺应性保持良好")

            print("\n综合建议：")
            for s in suggestions:
                print("- " + s)
        else:
            print("无法从非0类平均信号中提取有效IPPg特征。")
    else:
        print("\n当前画面中无非0类区域，局部指标无显著异常，无需特殊干预。")

    cv2.imshow("Classified Frame with Smooth Curves (B-spline)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def v_ratio(curr, base):
    # 一个辅助函数，用于计算相对于基线的比例
    if base == 0:
        return None
    return curr/base

if __name__ == '__main__':
    main()
