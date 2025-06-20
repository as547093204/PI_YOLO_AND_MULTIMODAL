#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import logging
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn import MultiheadAttention
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# ─── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal IPPG Ablation Experiments")
    parser.add_argument(
        '--exp_type',
        choices=[
            'all','full','no_temp_att','no_feat_att','no_all_att',
            'cnn_only','lstm_only','hc_only'
        ],
        default='full',
        help='Which variant to run (or "all" for batch run)'
    )
    return parser.parse_args()

# ─── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_directories():
    for d in [
        'output/mutimodal/attention',
        'output/mutimodal/metrics',
        'output/mutimodal/plots',
        'models/mutimodal',
        'logs/mutimodal'
    ]:
        os.makedirs(d, exist_ok=True)

def setup_logging():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(f'logs/mutimodal/training_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[fh, logging.StreamHandler()]
    )
    return ts

# ─── Data Loading ───────────────────────────────────────────────────────────────

class SignalDataset(Dataset):
    def __init__(self, signals, labels, augment=False):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = self.signals[idx]
        lbl = self.labels[idx]
        if self.augment:
            sig = sig + torch.randn_like(sig) * 0.05
            sig = sig * np.random.uniform(0.9, 1.1)
        return sig, lbl

def load_and_preprocess_data():
    # Assumes data/ulcer_signal_0.xlsx … ulcer_signal_4.xlsx exist
    signals, labels = [], []
    for i in range(5):
        arr = pd.read_excel(f'data/ulcer_signal_{i}.xlsx').values
        signals.append(arr)
        labels.append(np.full(len(arr), i))
    X = np.vstack(signals)
    y = np.hstack(labels)
    X = StandardScaler().fit_transform(X)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw = torch.FloatTensor(cw)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test, cw

# ─── Model Components ──────────────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        ctx = (x * w).sum(1)
        return ctx, w

class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        out, w = self.attn(x, x, x, need_weights=True)
        return self.norm(x + out), w

class FeatureExtractor(nn.Module):
    def __init__(self, sr=30):
        super().__init__()
        self.sr = sr

    def forward(self, x):
        feats = []
        for sig in x:
            mx, mn = sig.max(), sig.min()
            t = torch.tensor([mx, mx-mn, sig.var(), sig.std()], device=sig.device)
            fftm = torch.abs(torch.fft.fft(sig)).mean()
            f = torch.tensor([fftm], device=sig.device)
            s = sig.detach().cpu().numpy()
            peaks = [i for i in range(1, len(s)-1) if s[i]>s[i-1] and s[i]>s[i+1]]
            vals  = [i for i in range(1, len(s)-1) if s[i]<s[i-1] and s[i]<s[i+1]]
            if len(peaks)<1 or len(vals)<2:
                ippg = torch.zeros(5, device=sig.device)
            else:
                PA = torch.trapz(sig[vals[0]:vals[-1]], dx=1/self.sr)
                A2 = torch.trapz(sig[peaks[0]:vals[1]], dx=1/self.sr)
                PH = sig[peaks[0]] - sig[vals[0]]
                A1 = torch.trapz(sig[vals[0]:peaks[0]], dx=1/self.sr)
                hh = (PH + sig[vals[0]]) / 2
                li = next((i for i in range(vals[0], peaks[0]) if sig[i]>=hh), vals[0])
                ri = next((i for i in range(peaks[0], vals[0], -1) if sig[i]>=hh), peaks[0])
                PWHH = (ri - li) / self.sr
                ippg = torch.tensor([PA, A2, PH, A1, PWHH], device=sig.device)
            feats.append(torch.cat([t, f, ippg]))
        return torch.stack(feats)

class ImprovedFusionModel(nn.Module):
    def __init__(self, seq_len, num_classes,
                 use_cnn, use_lstm, use_hc,
                 use_temp_att, use_feat_att):
        super().__init__()
        self.use_cnn      = use_cnn
        self.use_lstm     = use_lstm
        self.use_hc       = use_hc
        self.use_temp_att = use_temp_att
        self.use_feat_att = use_feat_att

        # Handcrafted features
        if use_hc:
            self.fe = FeatureExtractor()
            self.static_fc = nn.Sequential(
                nn.Linear(10, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 64)
            )

        # CNN feature extractor
        if use_cnn:
            self.conv1 = nn.Conv1d(1,   32, 3, padding=1)
            self.bn1   = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32,  64, 3, padding=1)
            self.bn2   = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 256, 3, padding=1)
            self.bn3   = nn.BatchNorm1d(256)
            lstm_in = 256
        else:
            lstm_in = 1

        # Temporal self-attention
        if use_temp_att:
            self.ta = TemporalSelfAttention(lstm_in, heads=4)

        # LSTM
        if use_lstm:
            self.lstm = nn.LSTM(
                lstm_in, 512, num_layers=3,
                batch_first=True, bidirectional=True, dropout=0.5
            )

        # Feature-level attention
        if use_feat_att:
            dim = 1024 if use_lstm else lstm_in
            self.fa = AttentionLayer(dim)

        # Final head
        head_in = 0
        if use_lstm:
            head_in += 1024
        if use_hc:
            head_in += 64

        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        parts = []

        # ── 手工特征分支 ──
        if self.use_hc:
            sf = self.fe(x)                   # shape [B,10]
            sf = self.static_fc(sf)           # shape [B,64]
            parts.append(sf)

        # ── 如果启用了 CNN 或 LSTM，则走时序分支 ──
        if self.use_cnn or self.use_lstm:
            # CNN 特征
            if self.use_cnn:
                t = x.unsqueeze(1)            # [B,1,seq]
                t = torch.relu(self.bn1(self.conv1(t)))
                t = torch.relu(self.bn2(self.conv2(t)))
                t = torch.relu(self.bn3(self.conv3(t)))
                t = t.transpose(1, 2)         # [B, seq, 256]
            else:
                t = x.unsqueeze(-1)          # [B, seq, 1]

            # Temporal Attention
            if self.use_temp_att:
                t, _ = self.ta(t)

            # LSTM
            if self.use_lstm:
                t, _ = self.lstm(t)

            # Feature-Level Attention or 平均
            if self.use_feat_att:
                t, _ = self.fa(t)
            else:
                t = t.mean(dim=1)           # [B, hidden]

            parts.append(t)                 # 只有启用时序分支才加

        # ── 拼接并输出 ──
        combined = torch.cat(parts, dim=1)  # dims: [B, 64] 或 [B, 64+hidden] 等
        return self.head(combined)


# ─── Plotting Utilities ────────────────────────────────────────────────────────

def visualize_attention(model, loader, device, path):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x.to(device)
        _ = model(x)
        if hasattr(model, 'ta') and model.use_temp_att:
            ta_w = model.ta.attn.in_proj_weight  # placeholder
        # implement similar for feature attention if desired
    # (Omitted for brevity; reuse your existing visualize_attention code)

def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(path); plt.close()

import numpy as np
from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_prob, path):
    """
    如果 y_prob 是一维（或只有一个概率列），则按二分类绘制；
    否则按多分类（One-vs-Rest）绘制多个 ROC 曲线。
    """
    y_prob = np.array(y_prob)
    plt.figure(figsize=(6, 5))

    # 二分类情形
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        # 如果 y_prob 是二维但第二维为1，则取第一列
        prob = y_prob.ravel()
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
    else:
        # 多分类情形
        n_classes = y_prob.shape[1]
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], '--', color='gray')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metrics(train_losses, train_accs, val_losses, val_accs,
                 test_loss, test_acc, exp_type, ts):
    plt.figure(figsize=(12,5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val   Loss')
    if test_loss is not None:
        plt.axhline(test_loss, linestyle='--', color='r', label='Test Loss')
    plt.legend(); plt.title('Loss History')
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs,   label='Val   Acc')
    if test_acc is not None:
        plt.axhline(test_acc, linestyle='--', color='r', label='Test Acc')
    plt.legend(); plt.title('Accuracy History')
    plt.tight_layout()
    plt.savefig(f'output/mutimodal/plots/metrics_{exp_type}_{ts}.png')
    plt.close()

# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_one(model, train_loader, val_loader,
              criterion, optimizer, scheduler,
              device, epochs, exp_type, ts):
    train_losses, train_accs = [], []
    val_losses, val_accs     = [], []
    best_val_acc = 0
    early_patience = 250
    no_imp = 0

    epoch_pbar = tqdm(range(epochs),
                     desc=f"Epochs[{exp_type}]",
                     ncols=100)

    for ep in epoch_pbar:
        # --- training ---
        model.train()
        running_loss=0; corr=0; tot=0
        train_pbar = tqdm(train_loader,
                          desc=f"[{exp_type}] Train Ep{ep+1}/{epochs}",
                          leave=False, ncols=100)
        for x,y in train_pbar:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()

            running_loss += loss.item()
            _,pred = out.max(1)
            corr += (pred==y).sum().item()
            tot += y.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}",
                                   acc=f"{100*corr/tot:.2f}%")

        t_loss = running_loss/len(train_loader)
        t_acc  = 100*corr/tot
        train_losses.append(t_loss)
        train_accs.append(t_acc)

        # --- validation ---
        model.eval()
        vloss=0; corr=0; tot=0
        val_pbar = tqdm(val_loader,
                        desc=f"[{exp_type}] Val   Ep{ep+1}/{epochs}",
                        leave=False, ncols=100)
        with torch.no_grad():
            for x,y in val_pbar:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                vloss += loss.item()
                _,pred = out.max(1)
                corr += (pred==y).sum().item()
                tot += y.size(0)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}",
                                     acc=f"{100*corr/tot:.2f}%")

        v_loss = vloss/len(val_loader)
        v_acc  = 100*corr/tot
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        scheduler.step(v_loss)
        epoch_pbar.set_postfix(train_loss=f"{t_loss:.4f}",
                               train_acc =f"{t_acc:.2f}%",
                               val_loss=f"{v_loss:.4f}",
                               val_acc =f"{v_acc:.2f}%")

        # save best
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            no_imp = 0
            torch.save(model.state_dict(),
                       f"models/mutimodal/best_{exp_type}_{ts}.pth")
        else:
            no_imp += 1
            if no_imp > early_patience:
                break

        # periodic plots
        if (ep+1) % 10 == 0:
            visualize_attention(model, val_loader, device,
                                f"output/mutimodal/attention/attn_{exp_type}_{ep+1}_{ts}.png")
            plot_metrics(train_losses, train_accs,
                         val_losses, val_accs,
                         None, None, exp_type, ts)

    return {
        'train_losses': train_losses,
        'train_accs':   train_accs,
        'val_losses':   val_losses,
        'val_accs':     val_accs
    }

def evaluate(model, loader, criterion, device):
    model.eval()
    tloss=0; corr=0; tot=0
    preds=[]; labels=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            tloss += loss.item()
            p = torch.softmax(out,1)
            _,pmax = out.max(1)
            corr += (pmax==y).sum().item()
            tot += y.size(0)
            preds.extend(pmax.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())
    return tloss/len(loader), 100*corr/tot, labels, preds, np.array(probs)

def run_single_experiment(args, data, loaders, cw, device, ts):
    # unpack first 6 elements; cw separately
    X_train, X_val, X_test, y_train, y_val, y_test = data[:6]
    train_loader, val_loader, test_loader = loaders

    # flags adjustment to keep CNN for lstm_only
    flags = {
        'use_cnn': args.exp_type not in ['hc_only'],
        'use_lstm': args.exp_type not in ['hc_only'],
        'use_hc': args.exp_type in ['full', 'no_temp_att', 'no_feat_att', 'no_all_att', 'hc_only'],
        'use_temp_att': args.exp_type == 'full',
        'use_feat_att': args.exp_type == 'full',
    }

    model = ImprovedFusionModel(
        seq_len=X_train.shape[1],
        num_classes=5,
        **flags
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    writer = SummaryWriter(log_dir=f'logs/mutimodal/{ts}')

    logging.info(f"Running experiment: {args.exp_type}")
    history = train_one(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        device, epochs=1500,
        exp_type=args.exp_type, ts=ts
    )
    writer.close()

    # final evaluation
    model.load_state_dict(torch.load(f"models/mutimodal/best_{args.exp_type}_{ts}.pth", map_location=device))
    test_loss, test_acc, labels, preds, probs = evaluate(
        model, test_loader, criterion, device
    )
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec  = recall_score   (labels, preds, average='macro', zero_division=0)
    f1   = f1_score       (labels, preds, average='macro', zero_division=0)

    # append results
    res = {
        'exp_type':  args.exp_type,
        'test_loss': test_loss,
        'test_acc':  test_acc,
        'precision': prec,
        'recall':    rec,
        'f1_score':  f1,
        'timestamp': ts
    }
    df = pd.DataFrame([res])
    out_csv = 'output/mutimodal/all_experiments_results.csv'
    df.to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)

    # plots & reports
    plot_metrics(history['train_losses'], history['train_accs'],
                 history['val_losses'],    history['val_accs'],
                 test_loss, test_acc,
                 args.exp_type, ts)
    plot_confusion_matrix(labels, preds,
                          f'output/mutimodal/metrics/cm_{args.exp_type}_{ts}.png')
    plot_roc_curve(labels, probs,
                   f'output/mutimodal/metrics/roc_{args.exp_type}_{ts}.png')
    report = classification_report(labels, preds, target_names=[f'Class{i}' for i in range(5)])
    with open(f'output/mutimodal/metrics/report_{args.exp_type}_{ts}.txt','w') as f:
        f.write(report)
    logging.info(f"Experiment {args.exp_type} completed.")

def main():
    args = parse_args()
    set_seed(42)
    setup_directories()
    ts = setup_logging()

    data = load_and_preprocess_data()
    loaders = (
        DataLoader(SignalDataset(data[0], data[3], augment=True),  batch_size=32, shuffle=True),
        DataLoader(SignalDataset(data[1], data[4], augment=False), batch_size=32, shuffle=False),
        DataLoader(SignalDataset(data[2], data[5], augment=False), batch_size=32, shuffle=False),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    variants = ['full','no_temp_att','no_feat_att','no_all_att',
                'cnn_only','lstm_only','hc_only']
    if args.exp_type == 'all':
        for v in variants:
            args.exp_type = v
            run_single_experiment(args, data, loaders, data[6], device, ts)
    else:
        run_single_experiment(args, data, loaders, data[6], device, ts)

if __name__ == '__main__':
    main()