"""
BCI EEG Classification with SVM
Features: band powers (delta/theta/alpha/beta_low/beta_high/gamma),
          spectral entropy, Hjorth parameters, basic time statistics,
          Petrosian fractal dimension, Higuchi fractal dimension (HFD),
          Sample Entropy (SampEn), differential entropy (DE),
          zero-crossing rate (ZCR).
Preprocessing: baseline drift removal + z-score per segment +
               1-40 Hz bandpass + 60 Hz notch (bandstop) filter.
LOSO + 簡單多數決投票後處理。
加入：訓練端小幅度 data augmentation（feature-level）。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import os
import glob
import warnings

from scipy.signal import butter, filtfilt, iirnotch, welch, detrend
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")


# ======================
# Config
# ======================

class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_113-2"

    # SVM parameters
    KERNEL = "rbf"
    C = 5.0
    GAMMA = "scale"
    MAX_ITER = -1
    PROBABILITY = True  # 開啟 probability 以支持 predict_proba 和 decision_function
    RANDOM_STATE = 42

    # Signal processing
    SAMPLING_RATE = 500
    SEGMENT_LENGTH = 4.0      # seconds
    OVERLAP_RATIO = 0.4       # segment overlap

    # Feature selection
    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 28    # 現在總特徵約 33 維，這裡設置選 24 維

    # Data augmentation（只作用在 X_train）
    USE_AUGMENTATION = False
    AUGMENT_TIMES = 1
    AUGMENT_NOISE_STD = 0.02
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
# Preprocessing: baseline + filters + z-score
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
# Feature extraction helpers
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
    k_max: maximum interval (通常 6~10 即可)
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
            # 標準 Higuchi 正規化
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
    - m: 模板維度 (通常 2)
    - r: 門檻，乘上 std
    - max_len: 為了效率，長度過長時先 downsample
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


def band_power(f, psd, f1, f2):
    """
    Integrate power between [f1, f2) Hz.
    """
    idx = (f >= f1) & (f < f2)
    if not np.any(idx):
        return 0.0
    power = np.trapz(psd[idx], f[idx])
    return power


# ======================
# Feature extraction main
# ======================

def extract_features(segment, fs=Config.SAMPLING_RATE):
    """
    Extract features from a preprocessed segment.

    Features:
        - Time-domain: mean, std, min, max, skew, kurtosis
        - Hjorth: activity, mobility, complexity
        - Petrosian fractal dimension (PFD)
        - Higuchi fractal dimension (HFD)
        - Zero-crossing rate (ZCR)
        - Sample Entropy (SampEn)
        - Spectral entropy (1-40 Hz)
        - Band powers (delta/theta/alpha/beta_low/beta_high/gamma, log-power)
        - Differential entropy (DE) per band (Gaussian assumption)
        - Band ratios + relative powers
    """
    eps = 1e-12

    # --- time-domain features ---
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    min_val = np.min(segment)
    max_val = np.max(segment)
    skew_val = skew(segment)
    kurt_val = kurtosis(segment)

    activity, mobility, complexity = hjorth_parameters(segment)
    pfd = petrosian_fd(segment)
    hfd = higuchi_fd(segment)
    sampen = sample_entropy(segment)

    # zero-crossing rate
    if len(segment) > 1:
        zcr = np.mean(segment[1:] * segment[:-1] < 0)
    else:
        zcr = 0.0

    time_feats = np.array([
        mean_val, std_val, min_val, max_val,
        skew_val, kurt_val,
        activity, mobility, complexity,
        pfd, hfd, zcr, sampen
    ])
    # → 13 維

    # --- frequency-domain features via Welch ---
    nperseg = min(len(segment) // 2, 256) if len(segment) >= 64 else len(segment)
    freqs, psd = welch(segment, fs, nperseg=nperseg)

    # spectral entropy (1-40 Hz)
    spec_ent = spectral_entropy(psd, freqs, 1.0, 40.0)

    # band powers
    delta = band_power(freqs, psd, 1.0, 4.0)
    theta = band_power(freqs, psd, 4.0, 8.0)
    alpha = band_power(freqs, psd, 8.0, 13.0)
    beta_low = band_power(freqs, psd, 13.0, 20.0)
    beta_high = band_power(freqs, psd, 20.0, 30.0)
    gamma = band_power(freqs, psd, 30.0, 40.0)

    # log band powers（避免數值範圍太大）
    delta_log = np.log(delta + eps)
    theta_log = np.log(theta + eps)
    alpha_log = np.log(alpha + eps)
    beta_low_log = np.log(beta_low + eps)
    beta_high_log = np.log(beta_high + eps)
    gamma_log = np.log(gamma + eps)

    # Differential Entropy (Gaussian assumption, nats)
    # DE = 0.5 * ln(2 * pi * e * variance) ; 這裡以 band power 近似 variance
    const = 2 * np.pi * np.e
    de_delta = 0.5 * np.log(const * (delta + eps))
    de_theta = 0.5 * np.log(const * (theta + eps))
    de_alpha = 0.5 * np.log(const * (alpha + eps))
    de_beta_low = 0.5 * np.log(const * (beta_low + eps))
    de_beta_high = 0.5 * np.log(const * (beta_high + eps))
    de_gamma = 0.5 * np.log(const * (gamma + eps))

    # total power & beta total
    total_power = delta + theta + alpha + beta_low + beta_high + gamma + eps
    beta_total = beta_low + beta_high + eps

    # ratio features
    alpha_beta_ratio = alpha / beta_total
    theta_alpha_ratio = theta / (alpha + eps)
    ta_beta_ratio = (theta + alpha) / beta_total

    # relative powers
    delta_rel = delta / total_power
    theta_rel = theta / total_power
    alpha_rel = alpha / total_power
    beta_low_rel = beta_low / total_power
    beta_high_rel = beta_high / total_power
    gamma_rel = gamma / total_power

    band_feats = np.array([
        # log powers (6)
        delta_log, theta_log, alpha_log,
        beta_low_log, beta_high_log, gamma_log,
        # DE per band (6)
        de_delta, de_theta, de_alpha,
        de_beta_low, de_beta_high, de_gamma,
        # ratios (3)
        alpha_beta_ratio, theta_alpha_ratio, ta_beta_ratio,
        # relative powers (6)
        delta_rel, theta_rel, alpha_rel,
        beta_low_rel, beta_high_rel, gamma_rel
    ])
    # → 6 + 6 + 3 + 6 = 21 維

    # 組合所有 feature
    feats = np.concatenate([time_feats, np.array([spec_ent]), band_feats])
    # 13 (time/HFD/SampEn/ZCR) + 1 (entropy) + 21 (band) = 35 維
    return feats


# ======================
# Data augmentation（feature-level, train only）
# ======================

def augment_features(X, y):
    """
    小幅度 data augmentation：
        - 對每筆 feature 向量做 amplitude scaling + Gaussian noise
        - 總共產生 (1 + AUGMENT_TIMES) 倍的訓練資料
    只在 training fold 使用，不會碰到 test fold。
    """
    if not Config.USE_AUGMENTATION or Config.AUGMENT_TIMES <= 0:
        return X, y

    rng = np.random.default_rng(Config.RANDOM_STATE)

    X = np.asarray(X)
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
# Classifier
# ======================

class BandFeatureSVMClassifier:
    """BCI classifier using SVM with extended band / time features."""

    def __init__(self):
        self.model = SVC(
            kernel=Config.KERNEL,
            C=Config.C,
            gamma=Config.GAMMA,
            max_iter=Config.MAX_ITER,
            probability=True,  # 開啟了 probability 以支持 predict_proba 和 decision_function
            random_state=Config.RANDOM_STATE,
            class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.loss_curve_ = []   # placeholder

        # === Feature selection 設定 ===
        num_features = 35  # extract_features 回傳的維度
        k_features = min(Config.N_FEATURES_SELECT, num_features)
        self.feature_selector = (
            SelectKBest(f_classif, k=k_features)
            if Config.FEATURE_SELECTION
            else None
        )

    def fit(self, X, y):
        # 1. 標準化 (global z-score on features)
        X_scaled = self.scaler.fit_transform(X)

        # 2. 特徵選擇
        if self.feature_selector is not None:
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled

        # 3. SVM 訓練
        self.model.fit(X_selected, y)
        return self

    def _transform(self, X):
        """Internal helper: scale + feature_select."""
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            return self.feature_selector.transform(X_scaled)
        return X_scaled

    def predict(self, X):
        X_selected = self._transform(X)
        return self.model.predict(X_selected)

    def predict_proba(self, X):
        """返回每个类别的预测概率（需要 probability=True）"""
        X_selected = self._transform(X)
        return self.model.predict_proba(X_selected)

    def decision_function(self, X):
        """
        回傳 SVM 的 decision function（margin），
        正負代表類別，絕對值代表信心度大小。
        """
        X_selected = self._transform(X)
        return self.model.decision_function(X_selected)

    def get_loss_curve(self):
        return self.loss_curve_


# ======================
# 後處理：多數決投票平滑
# ======================

def smooth_predictions(preds, window_size=31):
    """
    對連續 segment 的預測做簡單多數決平滑，減少單點亂飄。
    window_size 建議用奇數（3/5/7）。
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
    """Load data from all subjects and extract features."""
    all_features = []
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

        relax_features = []
        for seg in relax_segments:
            seg_f = preprocess_segment(seg, fs=Config.SAMPLING_RATE)
            relax_features.append(extract_features(seg_f, fs=Config.SAMPLING_RATE))
        relax_features = np.array(relax_features)

        focus_features = []
        for seg in focus_segments:
            seg_f = preprocess_segment(seg, fs=Config.SAMPLING_RATE)
            focus_features.append(extract_features(seg_f, fs=Config.SAMPLING_RATE))
        focus_features = np.array(focus_features)

        if relax_features.size == 0 or focus_features.size == 0:
            print(f"Warning: Feature extraction failed for {subject_id}, skip.")
            continue

        relax_labels = np.zeros(len(relax_features))  # 0 = relax
        focus_labels = np.ones(len(focus_features))   # 1 = focus

        subject_features = np.vstack([relax_features, focus_features])
        subject_labels = np.hstack([relax_labels, focus_labels])
        subject_ids = [subject_id] * len(subject_labels)

        all_features.append(subject_features)
        all_labels.append(subject_labels)
        all_subjects.extend(subject_ids)

    if not all_features:
        print("Error: No valid data found.")
        return None, None, None

    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    return X, y, all_subjects


def leave_one_subject_out_validation():
    print("Starting LOSO Cross-Validation with band-feature SVM...")

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
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # 訓練端 data augmentation
        X_train_aug, y_train_aug = augment_features(X_train, y_train)

        clf = BandFeatureSVMClassifier()
        clf.fit(X_train_aug, y_train_aug)
        y_pred = clf.predict(X_test)

        # 多數決平滑
        y_pred_smooth = smooth_predictions(y_pred, window_size=31)

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
    fig.suptitle("BCI Classifier (Band Features, SVM) - LOSO Results", fontsize=16)

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

    # Loss curves (這版沒真的算，留空訊息)
    valid_loss_curves = [lc for lc in results["loss_curves"] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]):
            axes[2].plot(loss_curve, alpha=0.7, label=f"S{i+1}")
        axes[2].set_title("Training Loss Curves (First 5 Subjects)")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No loss curves (SVM solver)",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Training Loss Curves")

    plt.tight_layout()
    plt.savefig("bci_results_band_svm.png", dpi=300, bbox_inches="tight")
    plt.show()


# ======================
# Main
# ======================

def main():
    print("BCI EEG Classification (Band Features + SVM + Augmentation + DE/ZCR + HFD + SampEn)")
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
    print("\nResults saved to 'bci_results_band_svm.png'")


if __name__ == "__main__":
    main()
