import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(x * attention_weights, dim=1)
        return context, attention_weights

class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(TemporalSelfAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        self.attn_weights = None

    def forward(self, x):
        attn_output, attn_weights = self.multihead_attn(x, x, x, need_weights=True)
        out = self.layer_norm(x + attn_output)
        self.attn_weights = attn_weights
        return out, attn_weights

class FeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=30):
        super(FeatureExtractor, self).__init__()
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

    def calculate_time_domain_features(self, signal):
        device = signal.device
        maximum = torch.max(signal)
        ptp = torch.max(signal) - torch.min(signal)
        variance = torch.var(signal)
        std = torch.std(signal)

        return torch.tensor([maximum, ptp, variance, std], device=device)

    def calculate_frequency_domain_features(self, signal):
        device = signal.device
        fft = torch.fft.fft(signal)
        fft_magnitude = torch.abs(fft)
        mean_amplitude = torch.mean(fft_magnitude)

        return torch.tensor([mean_amplitude], device=device)

    def calculate_ippg_features(self, signal):
        device = signal.device
        peaks, valleys = self.find_peaks_valleys(signal)

        if len(peaks) < 1 or len(valleys) < 2:
            return torch.zeros(5, device=device)

        PA = torch.trapz(signal[valleys[0]:valleys[-1]], dx=1 / self.sampling_rate)
        A2 = torch.trapz(signal[peaks[0]:valleys[1]], dx=1 / self.sampling_rate)
        PH = signal[peaks[0]] - signal[valleys[0]]
        A1 = torch.trapz(signal[valleys[0]:peaks[0]], dx=1 / self.sampling_rate)

        left_idx, right_idx = self.find_half_height_points(signal, peaks[0], valleys[0])
        PWHH = (right_idx - left_idx) / self.sampling_rate

        return torch.tensor([PA, A2, PH, A1, PWHH], device=device)

    def forward(self, x):
        device = x.device
        batch_features = []

        for signal in x:
            time_features = self.calculate_time_domain_features(signal)
            freq_features = self.calculate_frequency_domain_features(signal)
            ippg_features = self.calculate_ippg_features(signal)

            features = torch.cat([
                time_features,
                freq_features,
                ippg_features
            ])

            batch_features.append(features)

        batch_features = torch.stack(batch_features)
        return batch_features.to(device)

class ImprovedFusionModel(nn.Module):
    def __init__(self, temporal_length=150, num_classes=5):
        super(ImprovedFusionModel, self).__init__()

        # 特征提取
        self.feature_extractor = FeatureExtractor()

        # 增强的CNN特征提取
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # 时序注意力
        self.temporal_attention = TemporalSelfAttention(256, num_heads=8)

        # 增强的LSTM
        self.lstm = nn.LSTM(256, 512, num_layers=3, batch_first=True,
                            bidirectional=True, dropout=0.5)

        self.feature_attention = AttentionLayer(1024)  # 512 * 2 for bidirectional

        # 统计特征处理
        self.static_fc = nn.Sequential(
            nn.Linear(10, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64)
        )

        # 特征融合
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # 5分类输出
        )

        self.temporal_attention_weights = None
        self.feature_attention_weights = None

    def forward(self, x):
        # 特征提取
        statistical_features = self.feature_extractor(x)

        # CNN特征提取
        temp = x.unsqueeze(1)  # [batch_size, 1, sequence_length]
        temp = self.bn1(self.conv1(temp))
        temp = torch.relu(temp)
        temp = self.bn2(self.conv2(temp))
        temp = torch.relu(temp)
        temp = self.bn3(self.conv3(temp))
        temp = torch.relu(temp)

        # 注意力处理
        temp = temp.transpose(1, 2)  # [batch_size, seq_length, embed_dim]
        temp, self.temporal_attention_weights = self.temporal_attention(temp)

        # LSTM处理
        lstm_out, _ = self.lstm(temp)
        attended_features, self.feature_attention_weights = self.feature_attention(lstm_out)

        # 统计特征处理
        static = self.static_fc(statistical_features)

        # 特征融合
        combined = torch.cat((attended_features, static), dim=1)
        output = self.fusion_fc(combined)

        return output
