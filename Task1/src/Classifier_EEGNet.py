"""
BCI EEG Classification with EEGNet (PyTorch)
Input: preprocessed EEG segments (1 channel, 4 s @ 500 Hz -> 2000 samples)
Preprocessing: baseline drift removal + 60 Hz notch + 1–40 Hz bandpass + per-segment z-score.
Evaluation: LOSO + 簡單多數決投票平滑。
原本的 SVM 與手刻 feature pipeline 改為端到端 EEGNet。
"""

import os
import glob
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, iirnotch, welch, detrend
from scipy.stats import skew, kurtosis  # 保留，如果之後要用 feature-based 方法
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ======================
# Config
# ======================

class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_113-2"

    # Signal processing
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 4.0      # seconds
    OVERLAP_RATIO = 0.4       # segment overlap

    # EEGNet training hyper-parameters
    NUM_EPOCHS = 180 # 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # 3e-3 
    WEIGHT_DECAY = 1e-5   # 1e-5
    DROPOUT = 0.3
    PRINT_EVERY = 10
    RANDOM_STATE = 42
    USE_CUDA = True  # 若有 GPU 則自動啟用

    # Data augmentation（作用在 segment 上，可視需要開啟）
    USE_AUGMENTATION = True
    AUGMENT_TIMES = 1
    AUGMENT_NOISE_STD = 0.01
    AUGMENT_SCALE_MIN = 0.98
    AUGMENT_SCALE_MAX = 1.02


# ======================
# Data loading & segmentation
# ======================

def load_eeg_data(subject_path):
    """Load EEG data for a single subject."""
    relax_file = os.path.join(subject_path, "1.txt")
    focus_file = os.path.join(subject_path, "2.txt")

    try:
        relax_data = np.loadtxt(relax_file)
        focus_data = np.loadtxt(focus_file)
        return relax_data, focus_data
    except Exception as e:
        print(f"Error loading data for {subject_path}: {e}")
        return None, None


def create_segments(data, segment_length_samples, overlap_samples):
    """Split continuous EEG signal into overlapping segments."""
    if len(data) < segment_length_samples:
        return np.array([])

    segments = []
    start = 0
    step = segment_length_samples - overlap_samples

    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]
        segments.append(segment)
        start += step

    return np.array(segments)


# ======================
# Preprocessing
# ======================

def preprocess_segment(segment, fs=Config.SAMPLING_RATE):
    """
    Remove baseline drift, apply 60 Hz notch + 1-40 Hz bandpass,
    then per-segment z-score normalization.
    """
    # 0. baseline drift removal (linear detrend)
    seg = detrend(segment, type="linear")

    # 1. 60 Hz notch (bandstop)
    Q = 30.0
    w0 = 60.0 / (fs / 2.0)          # normalized frequency (0~1, Nyquist = 1)
    b_notch, a_notch = iirnotch(w0, Q)
    seg = filtfilt(b_notch, a_notch, seg)

    # 2. 1–40 Hz bandpass
    low, high = 1.0, 40.0
    b_bp, a_bp = butter(4, [low / (fs / 2.0), high / (fs / 2.0)], btype="band")
    seg = filtfilt(b_bp, a_bp, seg)

    # 3. per-segment z-score normalization
    eps = 1e-12
    seg_mean = np.mean(seg)
    seg_std = np.std(seg)
    seg = (seg - seg_mean) / (seg_std + eps)

    return seg


# ======================
# （選擇性）特徵擴充 helper
#   目前 EEGNet 版本預設「不用」這些手刻 feature，
#   如果之後想 hybrid (feature + CNN)，可以直接取用。
# ======================

def hjorth_parameters(x):
    """
    Hjorth parameters: activity, mobility, complexity.
    """
    x = x - np.mean(x)
    eps = 1e-12

    var0 = np.var(x)
    dx = np.diff(x)
    var1 = np.var(dx)
    ddx = np.diff(dx)
    var2 = np.var(ddx)

    activity = var0
    mobility = np.sqrt(var1 / (var0 + eps))
    complexity = np.sqrt(var2 / (var1 + eps)) / (mobility + eps)

    return activity, mobility, complexity


def spectral_entropy(psd, freqs, fmin=1.0, fmax=40.0):
    """
    Normalized spectral entropy in the given band (0~1).
    """
    eps = 1e-12
    idx = (freqs >= fmin) & (freqs <= fmax)
    psd_band = psd[idx]
    if psd_band.size == 0:
        return 0.0

    psd_sum = np.sum(psd_band)
    if psd_sum <= 0:
        return 0.0

    psd_norm = psd_band / psd_sum
    ent = -np.sum(psd_norm * np.log2(psd_norm + eps))
    ent_norm = ent / np.log2(psd_norm.size)
    return ent_norm


def petrosian_fd(x):
    """
    Petrosian Fractal Dimension (PFD) of 1D signal x.
    """
    x = np.asarray(x)
    N = len(x)
    if N < 2:
        return 0.0

    diff = np.diff(x)
    # zero-crossings of the derivative
    N_delta = np.sum(diff[:-1] * diff[1:] < 0)
    if N_delta == 0:
        return 0.0

    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))


def higuchi_fd(x, k_max=8):
    """
    Higuchi Fractal Dimension (HFD) of 1D signal x.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < k_max * 2:
        return 0.0

    L = []
    for k in range(1, k_max + 1):
        Lk = 0.0
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                continue
            diff = np.abs(np.diff(x[idxs]))
            n_k = len(idxs)
            Lm = diff.sum() * (N - 1) / ((n_k - 1) * k)
            Lk += Lm
        Lk /= k
        L.append(Lk)

    L = np.array(L)
    lnL = np.log(L + 1e-12)
    lnk = np.log(1.0 / np.arange(1, k_max + 1))
    fd, _ = np.polyfit(lnk, lnL, 1)
    return fd


def sample_entropy(x, m=2, r=0.2, max_len=256):
    """
    Sample Entropy (SampEn) of 1D signal x.
    （保留以供未來使用，EEGNet 版本預設不用）
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N <= m + 1:
        return 0.0

    # downsample 以控制計算量
    if N > max_len:
        step = int(np.floor(N / max_len))
        step = max(step, 1)
        x = x[::step]
        N = len(x)
        if N <= m + 1:
            return 0.0

    sd = np.std(x)
    if sd < 1e-12:
        return 0.0

    r *= sd

    def _phi(order):
        Xm = np.array([x[i:i + order] for i in range(N - order + 1)])
        Nm = Xm.shape[0]
        count = 0.0
        for i in range(Nm - 1):
            dist = np.max(np.abs(Xm[i + 1:] - Xm[i]), axis=1)
            count += np.sum(dist <= r)
        return count

    B = _phi(m)
    A = _phi(m + 1)

    if B == 0 or A == 0:
        return 0.0

    return -np.log(A / B)


def wavelet_energy(segment, wavelet="db4", level=4):
    """
    Compute log-energy of wavelet detail coefficients for several levels.
    目前 EEGNet pipeline 沒使用，保留以供未來混合模型。
    """
    segment = np.asarray(segment, dtype=float)
    N = len(segment)
    if N < 2:
        return np.zeros(level, dtype=float)

    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(N, w.dec_len)
    if max_level <= 0:
        max_level = 1
    actual_level = min(level, max_level)

    coeffs = pywt.wavedec(segment, w, level=actual_level)
    energies = np.zeros(level, dtype=float)
    eps = 1e-12

    for i in range(actual_level):
        cD = coeffs[i + 1]
        e = np.sum(cD ** 2)
        energies[i] = np.log(e + eps)

    return energies


# ======================
# Data augmentation（segment-level）
# ======================

def augment_segments(X, y):
    """
    小幅度 data augmentation：
        - 對每筆 segment 做 amplitude scaling + Gaussian noise
        - 總共產生 (1 + AUGMENT_TIMES) 倍的訓練資料
    只在 training fold 使用，不會碰到 test fold。
    """
    if not Config.USE_AUGMENTATION or Config.AUGMENT_TIMES <= 0:
        return X, y

    rng = np.random.default_rng(Config.RANDOM_STATE)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(Config.AUGMENT_TIMES):
        scales = rng.uniform(
            Config.AUGMENT_SCALE_MIN,
            Config.AUGMENT_SCALE_MAX,
            size=(X.shape[0], 1),
        )
        std_per_sample = np.std(X, axis=1, keepdims=True) + 1e-12
        noise = rng.normal(
            loc=0.0,
            scale=Config.AUGMENT_NOISE_STD * std_per_sample,
            size=X.shape,
        )
        X_new = X * scales + noise
        X_aug_list.append(X_new)
        y_aug_list.append(y)

    X_all = np.vstack(X_aug_list)
    y_all = np.hstack(y_aug_list)
    return X_all, y_all


# ======================
# EEGNet model (PyTorch)
# ======================

class EEGNet(nn.Module):
    """
    簡化版 EEGNet（Lawhern et al., 2018），支援單通道輸入。
    Input shape: (batch, 1, chans, samples)
    這裡 chans 預設為 1（單導程），samples ≈ 2000。
    """

    def __init__(self, chans=1, samples=2000, num_classes=2,
                 F1=24, D=2, F2=48, kernel_length=64, kernel_length2=16,
                 dropout=0.3):

        super().__init__()
        self.chans = chans
        self.samples = samples

        # Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise conv across channels
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(chans, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Block 2: separable convolution
        self.separable_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, kernel_length2),
            groups=F1 * D,
            padding=(0, kernel_length2 // 2),
            bias=False,
        )
        self.separable_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # Global average pooling + classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, num_classes)

    def forward(self, x):
        # x: (batch, 1, chans, samples)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.global_avg_pool(x)      # (batch, F2, 1, 1)
        x = torch.flatten(x, 1)          # (batch, F2)
        x = self.classifier(x)           # (batch, num_classes)
        return x


# ======================
# Dataset wrapper
# ======================

class EEGSegmentDataset(Dataset):
    """
    X: [N, T] float32 numpy array (preprocessed segments)
    y: [N] int labels (0/1) or None for inference
    """

    def __init__(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.X = torch.from_numpy(X)  # [N, T]
        if y is None:
            self.y = None
        else:
            y = np.asarray(y, dtype=np.int64)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]          # [T]
        # reshape to (1, chans=1, samples=T)
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        if self.y is None:
            return x
        return x, self.y[idx]


# ======================
# EEGNet classifier wrapper（模擬原本 SVM classifier 介面）
# ======================

class EEGNetClassifier:
    """
    BCI classifier using EEGNet.
    介面提供 fit / predict / predict_proba / decision_function / get_loss_curve，
    方便和原本 SVM 版程式接軌。
    """

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda" if (Config.USE_CUDA and torch.cuda.is_available()) else "cpu"
        )
        self.model = None
        self.loss_curve_ = []

    def fit(self, X, y):
        """
        X: [N, T] numpy array (segments)
        y: [N] labels
        """
        torch.manual_seed(Config.RANDOM_STATE)
        np.random.seed(Config.RANDOM_STATE)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        #n_total = len(y)
        #n0 = np.sum(y == 0)  # relax
        #n1 = np.sum(y == 1)  # focus
#
        #w0 = n_total / (2.0 * n0)
        #w1 = n_total / (2.0 * n1)
        #class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(self.device)

        # 依照 Config 決定是否做 data augmentation
        X_aug, y_aug = augment_segments(X, y)

        dataset = EEGSegmentDataset(X_aug, y_aug)
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        _, seg_len = X.shape
        self.model = EEGNet(
            chans=1,
            samples=seg_len,
            num_classes=self.num_classes,
            dropout=Config.DROPOUT,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = 60,
            gamma = 0.1
        )

        self.loss_curve_ = []

        for epoch in range(Config.NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0
            total_samples = 0

            for batch in loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)      # [B, 1, 1, T]
                labels = labels.to(self.device)      # [B]

                optimizer.zero_grad()
                outputs = self.model(inputs)         # [B, num_classes]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / max(total_samples, 1)
            self.loss_curve_.append(epoch_loss)

            if (
                epoch == 0
                or (epoch + 1) % Config.PRINT_EVERY == 0
                or epoch == Config.NUM_EPOCHS - 1
            ):
                print(
                    f"  Epoch {epoch+1:3d}/{Config.NUM_EPOCHS}: "
                    f"train loss = {epoch_loss:.4f}"
                )
            scheduler.step()
        return self

    def _predict_logits(self, X):
        """
        Helper：回傳 logits，方便 predict / predict_proba / decision_function 共用。
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")

        self.model.eval()

        X = np.asarray(X, dtype=np.float32)
        dataset = EEGSegmentDataset(X, y=None)
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        logits_list = []
        with torch.no_grad():
            for batch in loader:
                # batch: [B, 1, 1, T]
                inputs = batch.to(self.device)
                outputs = self.model(inputs)   # [B, num_classes]
                logits_list.append(outputs.cpu().numpy())

        logits = np.concatenate(logits_list, axis=0)
        return logits

    def predict(self, X):
        logits = self._predict_logits(X)
        preds = np.argmax(logits, axis=1)
        return preds

    def predict_proba(self, X):
        """
        使用 softmax 將 logits 轉成機率。
        """
        logits = self._predict_logits(X)
        # softmax in numpy
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def decision_function(self, X):
        """
        對二元分類：回傳 positive-logit - negative-logit，
        多類別：直接回傳 logits。
        """
        logits = self._predict_logits(X)
        if logits.shape[1] == 2:
            return logits[:, 1] - logits[:, 0]
        return logits

    def get_loss_curve(self):
        return self.loss_curve_


# ======================
# 後處理：多數決投票平滑
# ======================

def smooth_predictions(preds, window_size=21):
    """
    對連續 segment 的預測做簡單多數決平滑，減少單點亂飄。
    window_size 建議用奇數（3/5/7/...）。
    """
    preds = np.asarray(preds).astype(int)
    n = len(preds)
    if window_size <= 1 or n == 0:
        return preds

    half = window_size // 2
    smoothed = preds.copy()

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = preds[start:end]
        vals, counts = np.unique(window, return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]

    return smoothed


# ======================
# LOSO evaluation
# ======================

def load_all_subjects():
    """
    Load data from all subjects and extract preprocessed segments.
    X: [N_segments, T]（T ≈ 2000）
    y: [N_segments]（0 = relax, 1 = focus）
    subjects: list of subject IDs per segment（之後做 LOSO）
    """
    all_segments = []
    all_labels = []
    all_subjects = []

    subject_folders = sorted(glob.glob(os.path.join(Config.DATASET_PATH, "S*")))
    if not subject_folders:
        print(f"Error: No subject folders found in {Config.DATASET_PATH}")
        return None, None, None

    seg_len = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
    overlap_samples = int(seg_len * Config.OVERLAP_RATIO)

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        relax_data, focus_data = load_eeg_data(subject_folder)
        if relax_data is None or focus_data is None:
            continue

        relax_segments = create_segments(relax_data, seg_len, overlap_samples)
        focus_segments = create_segments(focus_data, seg_len, overlap_samples)

        if relax_segments.size == 0 or focus_segments.size == 0:
            print(f"Warning: Not enough data to create segments for {subject_id}, skip.")
            continue

        # 對每個 segment 做 preprocessing（但不再抽 hand-crafted features）
        relax_proc = []
        for seg in relax_segments:
            seg_f = preprocess_segment(seg, fs=Config.SAMPLING_RATE)
            relax_proc.append(seg_f)
        relax_proc = np.array(relax_proc)  # [N_relax, T]

        focus_proc = []
        for seg in focus_segments:
            seg_f = preprocess_segment(seg, fs=Config.SAMPLING_RATE)
            focus_proc.append(seg_f)
        focus_proc = np.array(focus_proc)  # [N_focus, T]

        if relax_proc.size == 0 or focus_proc.size == 0:
            print(f"Warning: Preprocessing failed for {subject_id}, skip.")
            continue

        relax_labels = np.zeros(len(relax_proc), dtype=int)  # 0 = relax
        focus_labels = np.ones(len(focus_proc), dtype=int)   # 1 = focus

        subject_segments = np.vstack([relax_proc, focus_proc])   # [N_subj, T]
        subject_labels = np.hstack([relax_labels, focus_labels])
        subject_ids = [subject_id] * len(subject_labels)

        all_segments.append(subject_segments)
        all_labels.append(subject_labels)
        all_subjects.extend(subject_ids)

    if not all_segments:
        print("Error: No valid data found.")
        return None, None, None

    X = np.vstack(all_segments)
    y = np.hstack(all_labels)
    return X, y, all_subjects


def leave_one_subject_out_validation():
    print("Starting LOSO Cross-Validation with EEGNet...")

    X, y, subjects = load_all_subjects()
    if X is None:
        return None

    subjects = np.asarray(subjects)
    unique_subjects = sorted(list(set(subjects)))

    results = {
        "accuracies": [],
        "confusion_matrices": [],
        "loss_curves": [],
        "subject_names": []
    }

    for test_subject in unique_subjects:
        print(f"\n=== Testing on subject {test_subject} (LOSO) ===")
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        clf = EEGNetClassifier(num_classes=2)
        clf.fit(X_train, y_train)

        # 測試集預測
        y_pred = clf.predict(X_test)
        y_pred_smooth = smooth_predictions(y_pred, window_size=21)

        from sklearn.metrics import accuracy_score, confusion_matrix  # 僅此處使用
        acc = accuracy_score(y_test, y_pred_smooth)
        cm = confusion_matrix(y_test, y_pred_smooth, labels=[0, 1])

        results["accuracies"].append(acc)
        results["confusion_matrices"].append(cm)
        results["loss_curves"].append(clf.get_loss_curve())
        results["subject_names"].append(test_subject)

        print(f"{test_subject}: Accuracy = {acc:.3f}")

    return results


# ======================
# Plotting
# ======================

def plot_results(results):
    if results is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("BCI Classifier (EEGNet, raw segments) - LOSO Results", fontsize=16)

    # Accuracy by subject
    axes[0].bar(
        range(len(results["accuracies"])),
        results["accuracies"],
        color=[
            "green" if acc >= 0.7 else "orange" if acc >= 0.6 else "red"
            for acc in results["accuracies"]
        ],
    )
    axes[0].set_title("Accuracy by Subject")
    axes[0].set_xlabel("Subject Index")
    axes[0].set_ylabel("Accuracy")
    mean_acc = np.mean(results["accuracies"])
    axes[0].axhline(y=mean_acc, color="r", linestyle="--", label=f"Mean: {mean_acc:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Overall confusion matrix
    total_cm = np.sum(results["confusion_matrices"], axis=0)
    sns.heatmap(
        total_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Relax", "Focus"],
        yticklabels=["Relax", "Focus"],
        ax=axes[1],
    )
    axes[1].set_title("Overall Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # Training loss curves（最多畫前 5 個 subject）
    valid_loss_curves = [lc for lc in results["loss_curves"] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]):
            axes[2].plot(loss_curve, alpha=0.7, label=f"S{i+1}")
        axes[2].set_title("EEGNet Training Loss Curves (First 5 Subjects)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No loss curves",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Training Loss Curves")

    plt.tight_layout()
    plt.savefig("bci_results_eegnet_loso_2.png", dpi=300, bbox_inches="tight")
    plt.show()


# ======================
# Main
# ======================

def main():
    print("BCI EEG Classification (EEGNet, raw segments)")
    print("=" * 60)

    results = leave_one_subject_out_validation()
    if results is None:
        print("Validation failed.")
        return

    mean_acc = np.mean(results["accuracies"])
    std_acc = np.std(results["accuracies"])
    print(f"\nOverall Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    total_cm = np.sum(results["confusion_matrices"], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        relax_accuracy = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        focus_accuracy = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0

        relax_precision = total_cm[0, 0] / np.sum(total_cm[:, 0]) if np.sum(total_cm[:, 0]) > 0 else 0
        focus_precision = total_cm[1, 1] / np.sum(total_cm[:, 1]) if np.sum(total_cm[:, 1]) > 0 else 0

    print(f"\nRelax Class:")
    print(f"  - Accuracy (Recall): {relax_accuracy:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})")
    print(f"  - Precision:         {relax_precision:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[:, 0])})")

    print(f"\nConcentration Class:")
    print(f"  - Accuracy (Recall): {focus_accuracy:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})")
    print(f"  - Precision:         {focus_precision:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[:, 1])})")

    plot_results(results)
    print("\nResults saved to 'bci_results_eegnet_loso_2.png'")


if __name__ == "__main__":
    main()
